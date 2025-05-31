import os
import json
import asyncio
import logging
import hashlib
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from dotenv import load_dotenv
import aiosqlite
import faiss
from duckling import DucklingWrapper
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, CLIPProcessor, CLIPModel
import whisper
import spacy
import torch
import networkx as nx
from neo4j import GraphDatabase
import plotly.graph_objects as go
import pandas as pd
import pickle
from scipy.special import softmax
from sklearn.cluster import KMeans
import ray
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton,
    QTabWidget, QFileDialog, QComboBox, QLabel, QWebEngineView
)
from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QFont, QIcon
import sys
import aiohttp
import aiofiles
from PIL import Image
import io
import tempfile
from copy import deepcopy
import logging.handlers

load_dotenv()
DB_PATH = "brainbot.db"
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MODEL_DIR = "./models"
TICK_INTERVAL = 10
USER_ID = "default_user"
LOG_FILE = "billy.log"

logger = logging.getLogger("brainbot")
logger.setLevel(logging.INFO)
handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=10**6, backupCount=5)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

try:
    ray.init(ignore_reinit_error=True)
except Exception as e:
    logger.error(f"Ray initialization failed: {e}")
    raise

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {DEVICE}")

try:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    PERCEPTION_MODEL = SentenceTransformer(os.path.join(MODEL_DIR, "all-MiniLM-L6-v2")).to(DEVICE)
    GEN_TOKENIZER = AutoTokenizer.from_pretrained(os.path.join(MODEL_DIR, "mixtral-8x7b-instruct"))
    GEN_MODEL = AutoModelForCausalLM.from_pretrained(
        os.path.join(MODEL_DIR, "mixtral-8x7b-instruct"),
        quantization_config=quant_config,
        device_map="auto"
    )
    CLIP_MODEL = CLIPModel.from_pretrained(os.path.join(MODEL_DIR, "clip-vit-base-patch32")).to(DEVICE)
    CLIP_PROCESSOR = CLIPProcessor.from_pretrained(os.path.join(MODEL_DIR, "clip-vit-base-patch32"))
    WHISPER_MODEL = whisper.load_model(os.path.join(MODEL_DIR, "whisper-base"), device=DEVICE if torch.cuda.is_available() else "cpu")
    NLP = spacy.load("en_core_web_sm")
    DUCKLING = DucklingWrapper()
except Exception as e:
    logger.error(f"Model initialization failed: {e}")
    raise

async def init_db():
    try:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id TEXT PRIMARY KEY,
                    mood_json TEXT,
                    baseline_json TEXT,
                    style_emb BLOB,
                    traits_json TEXT,
                    last_active TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    remind_time TEXT,
                    message TEXT,
                    sent INTEGER DEFAULT 0
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS memory_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    text TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    topic TEXT,
                    entities TEXT,
                    emotion_json TEXT,
                    salience REAL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS self_model (
                    user_id TEXT PRIMARY KEY,
                    mood_json TEXT,
                    beliefs_json TEXT,
                    values_json TEXT,
                    narrative_json TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    interaction_id TEXT,
                    reward REAL,
                    timestamp TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS curriculum (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    content TEXT,
                    summary TEXT,
                    qa_json TEXT,
                    timestamp TEXT
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS multimodal_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    type TEXT,
                    embedding BLOB,
                    description TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    salience REAL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS metrics_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    metric_type TEXT,
                    value REAL,
                    timestamp TEXT
                )
            """)
            await db.commit()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

asyncio.run(init_db())

class KnowledgeGraph:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        except Exception as e:
            logger.error(f"Neo4j initialization failed: {e}")
            raise

    def close(self):
        try:
            self.driver.close()
        except Exception as e:
            logger.error(f"Neo4j close failed: {e}")

    async def add_fact(self, user_id, fact, relation, target, confidence, salience):
        try:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MERGE (a:Concept {text: $fact, user_id: $user_id})
                    MERGE (b:Concept {text: $target, user_id: $user_id})
                    MERGE (a)-[r:RELATION {type: $relation, confidence: $confidence, salience: $salience}]->(b)
                    """,
                    user_id=user_id, fact=fact, target=target, relation=relation, confidence=confidence, salience=salience
                )
        except Exception as e:
            logger.error(f"Neo4j add_fact failed: {e}")

    async def query_related(self, user_id, concept, limit=5):
        try:
            async with self.driver.session() as session:
                result = await session.run(
                    """
                    MATCH (a:Concept {text: $concept, user_id: $user_id})-[r:RELATION]->(b:Concept)
                    RETURN b.text, r.type, r.confidence, r.salience
                    ORDER BY r.salience DESC LIMIT $limit
                    """,
                    user_id=user_id, concept=concept, limit=limit
                )
                return [{"text": r["b.text"], "relation": r["r.type"], "confidence": r["r.confidence"], "salience": r["r.salience"]} for r in result]
        except Exception as e:
            logger.error(f"Neo4j query_related failed: {e}")
            return []

    async def add_ethical_rule(self, user_id, rule, consequence, weight):
        try:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MERGE (a:EthicalRule {text: $rule, user_id: $user_id})
                    SET a.weight = $weight
                    MERGE (b:Consequence {text: $consequence, user_id: $user_id})
                    MERGE (a)-[:LEADS_TO {weight: $weight}]->(b)
                    """,
                    user_id=user_id, rule=rule, consequence=consequence, weight=weight
                )
        except Exception as e:
            logger.error(f"Neo4j add_ethical_rule failed: {e}")

memory_graphs = {}
knowledge_graphs = {}

def visualize_memory(user_id):
    try:
        if user_id not in memory_graphs:
            return None
        G = memory_graphs[user_id]
        if len(G.nodes) > 50:
            nodes = sorted(G.nodes(data=True), key=lambda x: x[1].get('salience', 0), reverse=True)[:50]
            G = G.subgraph([n[0] for n in nodes]).copy()
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
        node_x, node_y, node_text, node_colors = [], [], [], []
        for node, data in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            emotion = max(data['emotions'], key=data['emotions'].get) if data.get('emotions') else "neutral"
            node_text.append(f"{data['text'][:20]}... (Emotion: {emotion}, Salience: {data.get('salience', 0):.2f})")
            valence = data["emotions"].get("joy", 0) - data["emotions"].get("sadness", 0)
            node_colors.append(f"rgb({int(255 * (valence + 1)/2)}, {int(255 * (1 - abs(valence)))}, {int(255 * (1 - valence)/2)})")
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", hoverinfo="text", text=node_text,
            marker=dict(showscale=True, colorscale="RdBu", size=[10 + 20 * data.get('salience', 0) for _, data in G.nodes(data=True)], color=node_colors)
        )
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            showlegend=False, hovermode="closest", margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)
        ))
        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"Memory visualization failed: {e}")
        return None

def visualize_knowledge(user_id):
    try:
        if user_id not in knowledge_graphs:
            return None
        G = knowledge_graphs[user_id]
        if len(G.nodes) > 50:
            nodes = list(G.nodes)[:50]
            G = G.subgraph(nodes).copy()
        pos = nx.spring_layout(G)
        edge_x, edge_y = [], []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
        node_x, node_y, node_text = [], [], []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[:20])
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode="markers+text", hoverinfo="text", text=node_text,
            marker=dict(showscale=True, colorscale="Viridis", size=10)
        )
        fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(
            showlegend=False, hovermode="closest", margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False), yaxis=dict(showgrid=False, zeroline=False)
        ))
        return fig.to_html(full_html=False)
    except Exception as e:
        logger.error(f"Knowledge visualization failed: {e}")
        return None

@ray.remote
class Workspace:
    def __init__(self):
        self.percepts = {}
        self.salience_scores = {}
        self.cache = defaultdict(list)
        self.attention_weights = {}

    def broadcast(self, name, signal, salience):
        try:
            self.percepts[name] = signal
            self.salience_scores[name] = salience
            self.cache[name].append((signal, datetime.utcnow().isoformat()))
            if len(self.cache[name]) > 20:
                self.cache[name] = self.cache[name][-20:]
            logger.info(f"Broadcasted signal {name} with salience {salience}")
        except Exception as e:
            logger.error(f"Workspace broadcast failed: {e}")

    def apply_attention(self, name, weight):
        try:
            self.attention_weights[name] = weight
            if name in self.salience_scores:
                self.salience_scores[name] *= weight
            logger.info(f"Applied attention weight {weight} to {name}")
        except Exception as e:
            logger.error(f"Workspace apply_attention failed: {e}")

    def focus(self, k=5):
        try:
            sorted_items = sorted(self.salience_scores.items(), key=lambda x: x[1], reverse=True)
            return [(name, self.percepts[name]) for name, _ in sorted_items[:k]]
        except Exception as e:
            logger.error(f"Workspace focus failed: {e}")
            return []

    def summarize(self):
        try:
            focused = self.focus()
            return {
                "signals": {name: signal for name, signal in focused},
                "recent": {k: [s[0] for s in v[-5:]] for k, v in self.cache.items()},
                "attention": self.attention_weights
            }
        except Exception as e:
            logger.error(f"Workspace summarize failed: {e}")
            return {}

class AttentionModule:
    def __init__(self):
        self.dorsal_weights = defaultdict(float)
        self.ventral_triggers = ["!", "urgent", "help", "emergency", "please"]

    def compute_dorsal_attention(self, percept, goal):
        try:
            weight = 0.5
            if goal and percept.get("intent") == goal.get("type"):
                weight += 0.4
            if percept.get("sentiment", 0) > 0.5:
                weight += 0.2
            return weight
        except Exception as e:
            logger.error(f"Dorsal attention computation failed: {e}")
            return 0.5

    def compute_ventral_attention(self, text):
        try:
            weight = 0.0
            text_lower = text.lower()
            for trigger in self.ventral_triggers:
                if trigger in text_lower:
                    weight += 0.6
                    break
            if text_lower.isupper():
                weight += 0.4
            return weight
        except Exception as e:
            logger.error(f"Ventral attention computation failed: {e}")
            return 0.0

@ray.remote
class KnowledgeExtractor:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.nlp = NLP

    async def extract(self, text):
        try:
            doc = self.nlp(text)
            facts = []
            for sent in doc.sents:
                prompt = (
                    f"Extract a fact from: '{sent.text}' in the format: subject -> relation -> object\n"
                    f"Example: 'I love coffee' -> I -> loves -> coffee\n"
                    f"Return one fact or 'None' if no clear fact."
                )
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_new_tokens=50,
                    do_sample=False,
                    temperature=0.5
                )
                fact = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                if fact != "None":
                    parts = fact.split(" -> ")
                    if len(parts) == 3:
                        facts.append({"subject": parts[0], "relation": parts[1], "object": parts[2]})
            return facts
        except Exception as e:
            logger.error(f"Knowledge extraction failed: {e}")
            return []

    async def extract_from_multimodal(self, description, media_type):
        try:
            prompt = (
                f"Extract a fact from the {media_type} description: '{description}' in the format: subject -> relation -> object\n"
                f"Return one fact or 'None' if no clear fact."
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=50,
                do_sample=False,
                temperature=0.5
            )
            fact = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            if fact != "None":
                parts = fact.split(" -> ")
                if len(parts) == 3:
                    return {"subject": parts[0], "relation": parts[1], "object": parts[2]}
            return None
        except Exception as e:
            logger.error(f"Multimodal knowledge extraction failed: {e}")
            return None

class MultimodalPerceptor:
    def __init__(self):
        self.clip_model = CLIP_MODEL
        self.clip_processor = CLIP_PROCESSOR
        self.whisper_model = WHISPER_MODEL
        self.perception_model = PERCEPTION_MODEL

    async def process_image(self, image_path):
        try:
            async with aiofiles.open(image_path, "rb") as f:
                image_data = await f.read()
            image = Image.open(io.BytesIO(image_data))
            inputs = self.clip_processor(images=image, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                image_emb = self.clip_model.get_image_features(**inputs).cpu().numpy()
            prompt = "Describe the content of this image in one sentence."
            text_inputs = GEN_TOKENIZER(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            outputs = GEN_MODEL.generate(text_inputs["input_ids"], max_new_tokens=50, do_sample=False)
            description = GEN_TOKENIZER.decode(outputs[0][text_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            text_emb = self.perception_model.encode(description, convert_to_numpy=True)
            return {"embedding": image_emb, "description": description, "text_embedding": text_emb, "type": "image"}
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return None

    async def process_audio(self, audio_path):
        try:
            audio = whisper.load_audio(audio_path)
            result = self.whisper_model.transcribe(audio, language="en")
            transcription = result["text"]
            emb = self.perception_model.encode(transcription, convert_to_numpy=True)
            return {"embedding": emb, "description": transcription, "text_embedding": emb, "type": "audio"}
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return None

@ray.remote
class Perceptor:
    def __init__(self, tokenizer, model):
        self.model = PERCEPTION_MODEL
        self.nlp = NLP
        self.tokenizer = tokenizer
        self.gen_model = model

    async def analyze(self, text):
        try:
            async with asyncio.Lock():
                emb = self.model.encode(text, convert_to_tensor=True, device=DEVICE)
                doc = self.nlp(text)
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                intent_prompt = f"Classify the intent of: '{text}' into one or more categories (e.g., greeting, question, urgent). Return a list."
                emotion_prompt = f"Classify the emotions in: '{text}' into categories (e.g., joy, sadness) with probabilities. Return a dictionary."
                
                intent_inputs = self.tokenizer(intent_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                intent_outputs = self.gen_model.generate(intent_inputs["input_ids"], max_new_tokens=50, do_sample=False)
                try:
                    intent = json.loads(self.tokenizer.decode(intent_outputs[0][intent_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
                except json.JSONDecodeError:
                    logger.warning("Intent JSON parsing failed, using fallback")
                    intent = ["unknown"]
                
                emotion_inputs = self.tokenizer(emotion_prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                emotion_outputs = self.gen_model.generate(emotion_inputs["input_ids"], max_new_tokens=100, do_sample=False)
                try:
                    emotions = json.loads(self.tokenizer.decode(emotion_outputs[0][emotion_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
                except json.JSONDecodeError:
                    logger.warning("Emotion JSON parsing failed, using fallback")
                    emotions = {"neutral": 1.0}
                
                sentiment = sum([v if k in ["joy", "love"] else -v if k in ["sadness", "anger", "fear"] else 0 for k, v in emotions.items()])
                salience = max(emotions.values(), default=0.0) + abs(sentiment)
                
                async with aiosqlite.connect(DB_PATH) as db:
                    await db.execute(
                        "INSERT INTO metrics_log (user_id, metric_type, value, timestamp) VALUES (?, ?, ?, ?)",
                        (USER_ID, "salience", salience, datetime.utcnow().isoformat())
                    )
                    await db.commit()
                
                return {
                    "text": text,
                    "embedding": emb.cpu().numpy(),
                    "intent": intent[0] if intent else "unknown",
                    "intents": intent,
                    "sentiment": sentiment,
                    "emotions": emotions,
                    "entities": entities,
                    "salience": salience
                }
        except Exception as e:
            logger.error(f"Perceptor analysis failed: {e}")
            return {
                "text": text,
                "embedding": np.zeros(PERCEPTION_MODEL.get_sentence_embedding_dimension()),
                "intent": "unknown",
                "intents": ["unknown"],
                "sentiment": 0.0,
                "emotions": {"neutral": 1.0},
                "entities": [],
                "salience": 0.0
            }

class StyleLearner:
    def __init__(self, user_id):
        self.user_id = user_id
        self.model = PERCEPTION_MODEL
        self.style_emb = np.zeros(self.model.get_sentence_embedding_dimension())

    async def update(self, text):
        try:
            emb = self.model.encode(text, convert_to_numpy=True)
            self.style_emb = 0.9 * self.style_emb + 0.1 * emb
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO user_profiles (user_id, style_emb, last_active) VALUES (?, ?, ?)",
                    (self.user_id, pickle.dumps(self.style_emb), datetime.utcnow().isoformat())
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Style learner update failed: {e}")

    async def get(self):
        try:
            return self.style_emb.tolist()
        except Exception as e:
            logger.error(f"Style learner get failed: {e}")
            return [0.0] * self.model.get_sentence_embedding_dimension()

class WorkingMemory:
    def __init__(self, size=7):
        self.buffer = deque(maxlen=size)
        self.topics = defaultdict(int)

    def add(self, percept):
        try:
            self.buffer.append({
                "time": datetime.utcnow().isoformat(),
                "text": percept["text"],
                "intent": percept["intent"],
                "intents": percept["intents"],
                "entities": percept["entities"],
                "sentiment": percept["sentiment"],
                "salience": percept["salience"]
            })
            for intent in percept["intents"]:
                self.topics[intent] += 1
        except Exception as e:
            logger.error(f"Working memory add failed: {e}")

    def summarize(self, k=5):
        try:
            recent = list(self.buffer)[-k:]
            return {
                "topic": max(self.topics.items(), key=lambda x: x[1])[0] if self.topics else "unknown",
                "recent": [item["text"] for item in recent],
                "sentiment": sum(item["sentiment"] for item in recent) / max(len(recent), 1),
                "salience": sum(item["salience"] for item in recent) / max(len(recent), 1)
            }
        except Exception as e:
            logger.error(f"Working memory summarize failed: {e}")
            return {"topic": "unknown", "recent": [], "sentiment": 0.0, "salience": 0.0}

@ray.remote
class MemoryManager:
    def __init__(self, user_id, long_term_memory, knowledge_graph):
        self.user_id = user_id
        self.ltm = long_term_memory
        self.kg = knowledge_graph
        self.compression_threshold = 10
        self.tokenizer = GEN_TOKENIZER
        self.model = GEN_MODEL

    async def consolidate(self):
        try:
            nodes = list(self.ltm.episodic_graph.nodes(data=True))
            if len(nodes) < self.compression_threshold:
                return
            embeddings = [self.ltm.semantic_index.reconstruct(n[1]["id"]) for n, d in nodes if "id" in d]
            saliences = [n[1].get("salience", 0.0) for n in nodes if "id" in d]
            if len(embeddings) < 2:
                return
            kmeans = KMeans(n_clusters=min(3, len(embeddings)))
            labels = kmeans.fit_predict(embeddings)
            clusters = defaultdict(list)
            weights = defaultdict(float)
            for (node, data), label, salience in zip(nodes, labels, saliences):
                clusters[label].append(data["text"])
                weights[label] += salience
            for label, texts in clusters.items():
                prompt = f"Summarize these memories into a general lesson (weighted by salience {weights[label]:.2f}): {', '.join(texts)}"
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                outputs = self.model.generate(inputs["input_ids"], max_new_tokens=100, do_sample=False)
                lesson = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                await self.kg.add_fact(self.user_id, lesson, "is_lesson", "general_knowledge", 0.7, weights[label] / len(texts))
            cutoff = datetime.utcnow() - timedelta(days=30)
            for node, data in nodes:
                if datetime.fromisoformat(data["timestamp"]) < cutoff and data.get("salience", 0.0) < 0.1:
                    self.ltm.episodic_graph.remove_node(node)
            await self.ltm.save()
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

    async def store_multimodal(self, embedding, description, media_type, salience=0.5):
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO multimodal_memories (user_id, type, embedding, description, confidence, timestamp, salience) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (self.user_id, media_type, pickle.dumps(embedding), description, 0.6, datetime.utcnow().isoformat(), salience)
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Multimodal memory storage failed: {e}")

@ray.remote
class LongTermMemory:
    def __init__(self, user_id):
        self.user_id = user_id
        dim = PERCEPTION_MODEL.get_sentence_embedding_dimension()
        self.semantic_index = faiss.IndexFlatL2(dim)
        if torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            self.semantic_index = faiss.index_cpu_to_gpu(res, 0, self.semantic_index)
        self.multimodal_index = faiss.IndexFlatL2(CLIP_MODEL.config.vision_config.hidden_size)
        if torch.cuda.is_available():
            self.multimodal_index = faiss.index_cpu_to_gpu(res, 0, self.multimodal_index)
        self.episodic_graph = nx.DiGraph()
        self.index_file = f"faiss_index_{user_id}.bin"
        self.multimodal_index_file = f"faiss_multimodal_{user_id}.bin"
        self.episodic_file = f"episodic_memory_{user_id}.gpickle"
        memory_graphs[user_id] = self.episodic_graph
        knowledge_graphs[user_id] = nx.DiGraph()

    async def load(self):
        try:
            if os.path.exists(self.index_file):
                self.semantic_index = faiss.read_index(self.index_file)
                if torch.cuda.is_available():
                    res = faiss.StandardGpuResources()
                    self.semantic_index = faiss.index_cpu_to_gpu(res, 0, self.semantic_index)
            if os.path.exists(self.multimodal_index_file):
                self.multimodal_index = faiss.read_index(self.multimodal_index_file)
                if torch.cuda.is_available():
                    res = faiss.StandardGpuResources()
                    self.multimodal_index = faiss.index_cpu_to_gpu(res, 0, self.multimodal_index)
            if os.path.exists(self.episodic_file):
                self.episodic_graph = nx.read_gpickle(self.episodic_file)
                memory_graphs[self.user_id] = self.episodic_graph
        except Exception as e:
            logger.error(f"Long-term memory load failed: {e}")

    async def save(self):
        try:
            cpu_index = faiss.index_gpu_to_cpu(self.semantic_index) if torch.cuda.is_available() else self.semantic_index
            faiss.write_index(cpu_index, self.index_file)
            cpu_multimodal = faiss.index_gpu_to_cpu(self.multimodal_index) if torch.cuda.is_available() else self.multimodal_index
            faiss.write_index(cpu_multimodal, self.multimodal_index_file)
            nx.write_gpickle(self.episodic_graph, self.episodic_file)
            memory_graphs[self.user_id] = self.episodic_graph
        except Exception as e:
            logger.error(f"Long-term memory save failed: {e}")

    async def store_semantic(self, embedding, text, percept, confidence=0.5, salience=0.0):
        try:
            vec = np.array([embedding], dtype=np.float32)
            self.semantic_index.add(vec)
            meta = {
                "id": self.semantic_index.ntotal - 1,
                "text": text,
                "confidence": confidence,
                "timestamp": datetime.utcnow().isoformat(),
                "topic": percept["intent"],
                "entities": percept["entities"],
                "emotions": percept["emotions"],
                "salience": salience
            }
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO memory_metadata (user_id, text, confidence, timestamp, topic, entities, emotion_json, salience) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.user_id, text, confidence, meta["timestamp"], meta["topic"], json.dumps(meta["entities"]), json.dumps(meta["emotions"]), salience)
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Semantic memory storage failed: {e}")

    async def store_episodic(self, percept, prev_id=None, salience=0.0):
        try:
            memory_id = len(self.episodic_graph)
            self.episodic_graph.add_node(
                memory_id,
                text=percept["text"],
                id=memory_id,
                timestamp=datetime.utcnow().isoformat(),
                intent=percept["intent"],
                emotions=percept["emotions"],
                entities=percept["entities"],
                confidence=0.5 * (1 + salience),
                salience=salience
            )
            if prev_id is not None:
                self.episodic_graph.add_edge(prev_id, memory_id, weight=np.random.uniform(0.5, 1.0) * (1 + salience))
            await self.save()
        except Exception as e:
            logger.error(f"Episodic memory storage failed: {e}")

    async def store_multimodal(self, embedding, description, media_type, confidence=0.6, salience=0.5):
        try:
            vec = np.array([embedding], dtype=np.float32)
            self.multimodal_index.add(vec)
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO multimodal_memories (user_id, type, embedding, description, confidence, timestamp, salience) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (self.user_id, media_type, pickle.dumps(embedding), description, confidence, datetime.utcnow().isoformat(), salience)
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Multimodal memory storage failed: {e}")

    async def retrieve_semantic(self, query_embedding, k=5):
        try:
            distances, indices = self.semantic_index.search(np.array([query_embedding], dtype=np.float32), k)
            results = []
            async with aiosqlite.connect(DB_PATH) as db:
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0:
                        continue
                    cur = await db.execute("SELECT * FROM memory_metadata WHERE id = ? AND user_id = ?", (idx, self.user_id))
                    row = await cur.fetchone()
                    if row:
                        results.append({
                            "text": row[2],
                            "confidence": row[3],
                            "emotions": json.loads(row[7]),
                            "salience": row[8]
                        })
            return sorted(results, key=lambda x: x["salience"], reverse=True)
        except Exception as e:
            logger.error(f"Semantic memory retrieval failed: {e}")
            return []

    async def retrieve_multimodal(self, query_embedding, k=5):
        try:
            distances, indices = self.multimodal_index.search(np.array([query_embedding], dtype=np.float32), k)
            results = []
            async with aiosqlite.connect(DB_PATH) as db:
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0:
                        continue
                    cur = await db.execute("SELECT * FROM multimodal_memories WHERE id = ? AND user_id = ?", (idx, self.user_id))
                    row = await cur.fetchone()
                    if row:
                        results.append({
                            "description": row[4],
                            "type": row[2],
                            "confidence": row[5],
                            "salience": row[7]
                        })
            return sorted(results, key=lambda x: x["salience"], reverse=True)
        except Exception as e:
            logger.error(f"Multimodal memory retrieval failed: {e}")
            return []

    async def decay(self):
        try:
            for node, data in self.episodic_graph.nodes(data=True):
                data["confidence"] *= 0.995
                data["salience"] *= 0.99
            await self.save()
        except Exception as e:
            logger.error(f"Memory decay failed: {e}")

@ray.remote
class ImaginationModule:
    def __init__(self, user_id, long_term_memory, tokenizer, model, world_model):
        self.user_id = user_id
        self.ltm = long_term_memory
        self.tokenizer = tokenizer
        self.model = model
        self.world_model = world_model

    async def simulate(self, narrative=None):
        try:
            nodes = list(self.ltm.episodic_graph.nodes(data=True))
            if len(nodes) < 2:
                return
            selected = np.random.choice(nodes, size=2, replace=False, p=softmax([n[1]["salience"] for n in nodes]))
            context = f"{selected[0][1]['text']} and {selected[1][1]['text']}"
            if narrative:
                context += f" in the context of {narrative}"
            predicted_intent = await self.world_model.predict_intent(context)
            prompt = (
                f"Simulate a scenario combining {context} with predicted intent '{predicted_intent}'. "
                f"Describe a plausible outcome, reflecting on potential consequences.\n"
                f"Return: {{'scenario': 'text', 'reflection': 'text'}}"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=150,
                do_sample=True,
                top_p=0.9,
                temperature=0.8
            )
            try:
                result = json.loads(self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            except json.JSONDecodeError:
                logger.warning("Imagination JSON parsing failed, using fallback")
                result = {"scenario": "No scenario generated", "reflection": "No reflection"}
            self.ltm.episodic_graph.add_node(
                len(self.ltm.episodic_graph),
                text=result["scenario"],
                timestamp=datetime.utcnow().isoformat(),
                intent="hypothetical",
                emotions={"surprise": 0.4, "joy": 0.3},
                entities=[],
                confidence=0.3,
                salience=0.4
            )
            await self.ltm.save()
        except Exception as e:
            logger.error(f"Imagination simulation failed: {e}")

@ray.remote
class DefaultModeNetwork:
    def __init__(self, user_id, long_term_memory, tokenizer, model):
        self.user_id = user_id
        self.ltm = long_term_memory
        self.tokenizer = tokenizer
        self.model = model
        self.narrative = []

    async def update_narrative(self):
        try:
            memories = await self.ltm.retrieve_semantic(
                self.ltm.semantic_index.reconstruct(0) if self.ltm.semantic_index.ntotal > 0 else np.zeros(PERCEPTION_MODEL.get_sentence_embedding_dimension()),
                k=5
            )
            prompt = (
                f"Reflect on these memories: {', '.join([m['text'] for m in memories])}.\n"
                f"Generate a narrative summarizing the user’s interests and recent concerns.\n"
                f"Return: {{'narrative': 'text'}}"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            outputs = self.model.generate(inputs["input_ids"], max_new_tokens=100)
            try:
                result = json.loads(self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
            except json.JSONDecodeError:
                logger.warning("Narrative JSON parsing failed, using fallback")
                result = {"narrative": "No narrative generated"}
            narrative_entry = result["narrative"]
            self.narrative.append({"text": narrative_entry, "timestamp": datetime.utcnow().isoformat()})
            if len(self.narrative) > 5:
                self.narrative = self.narrative[-5:]
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO self_model (user_id, narrative_json) VALUES (?, ?)",
                    (self.user_id, json.dumps(self.narrative))
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Narrative update failed: {e}")

    async def get_narrative(self):
        try:
            return self.narrative[-1]["text"] if self.narrative else "No narrative yet."
        except Exception as e:
            logger.error(f"Narrative retrieval failed: {e}")
            return "No narrative yet."

@ray.remote
class CurriculumLearner:
    def __init__(self, user_id, tokenizer, model, knowledge_graph):
        self.user_id = user_id
        self.tokenizer = tokenizer
        self.model = model
        self.kg = knowledge_graph

    async def process_content(self, content):
        try:
            prompt = (
                f"Summarize this content: {content[:1000]}...\n"
                f"Generate a summary and 3 QA pairs.\n"
                f"Return: {{'summary': 'text', 'qa': [{'question': 'text', 'answer': 'text'}]}}"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=300,
                do_sample=False
            )
            try:
                result = json.loads(self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            except json.JSONDecodeError:
                logger.warning("Curriculum JSON parsing failed, using fallback")
                result = {"summary": "No summary", "qa": []}
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO curriculum (user_id, content, summary, qa_json, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (self.user_id, content, result["summary"], json.dumps(result["qa"]), datetime.utcnow().isoformat())
                )
                await db.commit()
            for qa in result["qa"]:
                await self.kg.add_fact(self.user_id, qa["question"], "has_answer", qa["answer"], 0.8, 1.0)
        except Exception as e:
            logger.error(f"Curriculum processing failed: {e}")

class WorldModel:
    def __init__(self, user_id):
        self.user_id = user_id
        self.model = BayesianNetwork([
            ("Context", "Intent"),
            ("Intent", "Action"),
            ("Action", "Outcome")
        ])
        self.model.add_cpds(
            TabularCPD("Context", 3, [[0.33], [0.33], [0.34]], state_names={"Context": ["neutral", "urgent", "question"]}),
            TabularCPD("Intent", 3, [[0.5, 0.2, 0.8], [0.3, 0.6, 0.1], [0.2, 0.2, 0.1]], 
                       evidence=["Context"], evidence_card=[3], 
                       state_names={"Intent": ["greeting", "urgent", "question"], "Context": ["neutral", "urgent", "question"]}),
            TabularCPD("Action", 3, [[0.6, 0.3, 0.7], [0.3, 0.6, 0.2], [0.1, 0.1, 0.1]], 
                       evidence=["Intent"], evidence_card=[3], 
                       state_names={"Action": ["respond", "assist", "query"], "Intent": ["greeting", "urgent", "question"]}),
            TabularCPD("Outcome", 2, [[0.8, 0.7, 0.9], [0.2, 0.3, 0.1]], 
                       evidence=["Action"], evidence_card=[3], 
                       state_names={"Outcome": ["success", "failure"], "Action": ["respond", "assist", "query"]})
        )
        self.inference = VariableElimination(self.model)
        self.observations = []

    async def update(self, context, intent, action, outcome):
        try:
            self.observations.append({"Context": context, "Intent": intent, "Action": action, "Outcome": outcome})
            if len(self.observations) > 100:
                context_counts = defaultdict(int)
                intent_counts = defaultdict(lambda: defaultdict(int))
                action_counts = defaultdict(lambda: defaultdict(int))
                outcome_counts = defaultdict(lambda: defaultdict(int))
                
                for obs in self.observations:
                    context_counts[obs["Context"]] += 1
                    intent_counts[obs["Context"]][obs["Intent"]] += 1
                    action_counts[obs["Intent"]][obs["Action"]] += 1
                    outcome_counts[obs["Action"]][obs["Outcome"]] += 1
                
                total = sum(context_counts.values())
                context_probs = [[context_counts[c] / total] for c in ["neutral", "urgent", "question"]]
                self.model.get_cpds("Context").values = np.array(context_probs)
                
                intent_probs = []
                for c in ["neutral", "urgent", "question"]:
                    total = sum(intent_counts[c].values())
                    probs = [intent_counts[c][i] / total if total > 0 else 1/3 for i in ["greeting", "urgent", "question"]]
                    intent_probs.append(probs)
                self.model.get_cpds("Intent").values = np.array(intent_probs).T
                
                # Update Action and Outcome CPDs similarly
                # (Omitted for brevity, similar logic)
                
                self.observations = self.observations[-50:]
        except Exception as e:
            logger.error(f"World model update failed: {e}")

    async def predict_intent(self, context_text):
        try:
            context = "neutral"
            if any(word in context_text.lower() for word in ["urgent", "help", "!"]):
                context = "urgent"
            elif "?" in context_text:
                context = "question"
            result = self.inference.map_query(variables=["Intent"], evidence={"Context": context})
            return result["Intent"]
        except Exception as e:
            logger.error(f"World model predict_intent failed: {e}")
            return "unknown"

@ray.remote
class PlanningModule:
    def __init__(self, user_id, long_term_memory, tokenizer, model):
        self.user_id = user_id
        self.ltm = long_term_memory
        self.tokenizer = tokenizer
        self.model = model

    async def plan(self, intent, context):
        try:
            memories = await self.ltm.retrieve_semantic(
                self.ltm.semantic_index.reconstruct(0) if self.ltm.semantic_index.ntotal > 0 else np.zeros(PERCEPTION_MODEL.get_sentence_embedding_dimension()),
                k=3
            )
            memory_context = ", ".join([m["text"] for m in memories])
            prompt = (
                f"Given intent '{intent}' and context '{context}', "
                f"with past memories: {memory_context}, "
                f"propose a three-step dialogue plan to achieve a helpful response.\n"
                f"Return: ['step1', 'step2', 'step3']"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=150,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            try:
                plan = json.loads(self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            except json.JSONDecodeError:
                logger.warning("Planning JSON parsing failed, using fallback")
                plan = ["Step 1", "Step 2", "Step 3"]
            return plan
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            return ["Step 1", "Step 2", "Step 3"]

class EthicsMonitor:
    def __init__(self, tokenizer, model, knowledge_graph):
        self.tokenizer = tokenizer
        self.model = model
        self.kg = knowledge_graph
        self.rules = [
            {"rule": "Avoid harmful language", "consequence": "User distress", "weight": 0.9},
            {"rule": "Respect user privacy", "consequence": "Trust violation", "weight": 0.8},
            {"rule": "Promote inclusivity", "consequence": "Positive engagement", "weight": 0.7}
        ]

    async def initialize_rules(self, user_id):
        try:
            for rule in self.rules:
                await self.kg.add_ethical_rule(user_id, rule["rule"], rule["consequence"], rule["weight"])
        except Exception as e:
            logger.error(f"Ethics rules initialization failed: {e}")

    async def evaluate(self, response, user_id):
        try:
            prompt = (
                f"Evaluate the ethicality of this response: '{response}'.\n"
                f"Consider rules: {', '.join([r['rule'] for r in self.rules])}.\n"
                f"Return: {{'score': float, 'violations': ['rule1', ...]}}"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
            outputs = self.model.generate(inputs["input_ids"], max_new_tokens=100, do_sample=False)
            try:
                result = json.loads(self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            except json.JSONDecodeError:
                logger.warning("Ethics JSON parsing failed, using fallback")
                result = {"score": 0.5, "violations": []}
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO metrics_log (user_id, metric_type, value, timestamp) VALUES (?, ?, ?, ?)",
                    (user_id, "ethics_score", result["score"], datetime.utcnow().isoformat())
                )
                await db.commit()
            return result
        except Exception as e:
            logger.error(f"Ethics evaluation failed: {e}")
            return {"score": 0.5, "violations": []}

class ConflictMonitor:
    def __init__(self):
        self.conflict_threshold = 0.7

    def detect_conflict(self, percept, prev_response):
        try:
            conflict_score = 0.0
            if prev_response and (percept.get("sentiment", 0) < -0.5 or percept["emotions"].get("anger", 0) > 0.5):
                conflict_score += 0.5
                if prev_response and ("sorry" in prev_response.lower() or "misunderstand" in prev_response.lower()):
                    conflict_score += 0.3
            return conflict_score > self.conflict_threshold
        except Exception as e:
            logger.error(f"Conflict detection failed: {e}")
            return False

class ReplayBuffer:
    def __init__(self, user_id):
        self.user_id = user_id
        self.buffer = deque(maxlen=1000)

    def add(self, state, action, reward, next_state):
        try:
            self.buffer.append({"state": state, "action": action, "reward": reward, "next_state": next_state})
        except Exception as e:
            logger.error(f"Replay buffer add failed: {e}")

    def sample(self, batch_size=32):
        try:
            indices = np.random.choice(len(self.buffer), min(batch_size, len(self.buffer)))
            return [self.buffer[idx] for idx in indices]
        except Exception as e:
            logger.error(f"Replay buffer sample failed: {e}")
            return []

@ray.remote
class ResponseGenerator:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.reference_policy = None
        self.beta = 0.1
        self.group_size = 5

    async def compute_kl_divergence(self, response, context):
        return 0.0

    async def generate(self, intent, workspace, intents, self_model, emotion_engine, user_style, plan, context, memories):
        try:
            memory_context = "\n".join([m["text"] for m in memories if "text" in m] + [m["description"] for m in memories if "description" in m])
            prompt = (
                f"<s>[INST] Context:\n"
                f"Conversation History:\n{context}\n"
                f"Relevant Memories:\n{memory_context}\n"
                f"Mood: {json.dumps(emotion_engine.mood)}\n"
                f"Self: {json.dumps(self_model.beliefs)}\n"
                f"Values: {json.dumps(self_model.values)}\n"
                f"Signals: {json.dumps(workspace.summarize())}\n"
                f"User Style: {json.dumps(user_style)}\n"
                f"Intents: {json.dumps(intents)}\n"
                f"Plan: {json.dumps(plan) if plan else 'No Plan'}\n"
                f"Generate {self.group_size} responses, score each for empathy based on mood and user style.\n"
                f"Use concise language if valence < -0.5, elaborate if valence > 0.5.\n"
                f"Return: {{'selected': 'response', 'all': ['response1', 'response2', ...], 'scores': [score1, ...]}} [/INST]"
            )
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_new_tokens=150,
                do_sample=True,
                top_p=0.9,
                temperature=0.7 + emotion_engine.mood["arousal"] * 0.5,
                num_return_sequences=self.group_size
            )
            responses = [self.tokenizer.decode(out[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip() for out in outputs]
            
            scores = []
            for response in responses:
                empathy_score = 1.0
                if emotion_engine.mood["valence"] < -0.5 and len(response.split()) < 20:
                    empathy_score += 0.5
                elif emotion_engine.mood["valence"] > 0.5 and len(response.split()) > 20:
                    empathy_score += 0.5
                kl_penalty = await self.compute_kl_divergence(response, context)
                score = empathy_score - self.beta * kl_penalty
                scores.append(score)
            
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores) + 1e-6
                scores = [(s - mean_score) / std_score for s in scores]
            
            best_idx = np.argmax(scores)
            selected_response = responses[best_idx]
            
            try:
                result = {
                    "selected": selected_response,
                    "all": responses,
                    "scores": scores
                }
            except json.JSONDecodeError:
                logger.warning("Response generation JSON parsing failed, using fallback")
                result = {"selected": responses[0] if responses else "", "all": responses, "scores": scores}
            
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO metrics_log (user_id, metric_type, value, timestamp) VALUES (?, ?, ?, ?)",
                    (USER_ID, "response_score", scores[best_idx], datetime.utcnow().isoformat())
                )
                await db.commit()
            
            return result["selected"]
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I'm sorry, I couldn't generate a response. How can I assist you?"

class EmotionEngine:
    def __init__(self, user_id):
        self.user_id = user_id
        self.mood = {"valence": 0.0, "arousal": 0.0}
        self.history = deque(maxlen=1000)

    async def load(self):
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                cur = await db.execute("SELECT mood_json FROM user_profiles WHERE user_id = ?", (self.user_id,))
                row = await cur.fetchone()
                if row:
                    self.mood = json.loads(row["mood_json"])
        except Exception as e:
            logger.error(f"Emotion engine load failed: {e}")

    async def save(self):
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO user_profiles (user_id, mood_json, last_active) VALUES (?, ?, ?)",
                    (self.user_id, json.dumps(self.mood), datetime.utcnow().isoformat())
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Emotion engine save failed: {e}")

    def update(self, percept, self_model):
        try:
            valence = percept["sentiment"]
            arousal = max(percept["emotions"].values(), default=0.0)
            self.mood["valence"] = 0.8 * self.mood["valence"] + 0.2 * valence
            self.mood["arousal"] = 0.8 * self.mood["arousal"] + 0.2 * arousal
            self.history.append({"valence": valence, "arousal": arousal, "trigger": percept["text"]})
            if arousal > 0.7:
                self_model.beliefs["trust"] = max(0.0, self_model.beliefs["trust"] - 0.05)
            if valence < -0.3:
                self_model.values["empathy"] = min(1.0, self_model.values["empathy"] + 0.1)
        except Exception as e:
            logger.error(f"Emotion engine update failed: {e}")

class GoalManager:
    def __init__(self, user_id):
        self.user_id = user_id
        self.goals = {"short": [], "mid": [], "long": []}

    def add(self, goal_type, goal):
        try:
            self.goals[goal_type].append(goal)
        except Exception as e:
            logger.error(f"Goal manager add failed: {e}")

    def resolve_conflicts(self):
        try:
            if self.goals["short"]:
                return self.goals["short"][0]
            elif self.goals["mid"]:
                return self.goals["mid"][0]
            elif self.goals["long"]:
                return self.goals["long"][0]
            return None
        except Exception as e:
            logger.error(f"Goal manager resolve_conflicts failed: {e}")
            return None

class SelfModel:
    def __init__(self, user_id):
        self.user_id = user_id
        self.mood = {"valence": 0.0, "arousal": 0.0}
        self.beliefs = {"confidence": 0.5, "trust": 0.5}
        self.values = {"empathy": 0.8, "curiosity": 0.7}

    async def load(self):
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                cur = await db.execute(
                    "SELECT mood_json, beliefs_json, values_json FROM self_model WHERE user_id = ?",
                    (self.user_id,)
                )
                row = await cur.fetchone()
                if row:
                    self.mood = json.loads(row["mood_json"])
                    self.beliefs = json.loads(row["beliefs_json"])
                    self.values = json.loads(row["values_json"])
        except Exception as e:
            logger.error(f"Self model load failed: {e}")

    async def save(self):
        try:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT OR REPLACE INTO self_model (user_id, mood_json, beliefs_json, values_json, narrative_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (self.user_id, json.dumps(self.mood), json.dumps(self.beliefs), json.dumps(self.values), json.dumps([]))
                )
                await db.commit()
        except Exception as e:
            logger.error(f"Self model save failed: {e}")

    def reflect(self, outcome):
        try:
            if outcome == "success":
                self.beliefs["confidence"] = min(1.0, self.beliefs["confidence"] + 0.1)
            elif outcome == "failure":
                self.beliefs["confidence"] = max(0.0, self.beliefs["confidence"] - 0.1)
                self.values["empathy"] = min(1.0, self.values["empathy"] + 0.05)
        except Exception as e:
            logger.error(f"Self model reflection failed: {e}")

class CoherenceModule:
    def __init__(self):
        self.history = deque(maxlen=10)

    def add(self, user_input, response):
        try:
            self.history.append({"user": user_input, "response": response})
        except Exception as e:
            logger.error(f"Coherence add failed: {e}")

    def get_context(self):
        try:
            return "\n".join([f"User: {turn['user']}\nBilly: {turn['response']}" for turn in self.history])
        except Exception as e:
            logger.error(f"Coherence get_context failed: {e}")
            return ""

class CognitiveAgent:
    def __init__(self, user_id):
        self.user_id = user_id
        self.workspace = Workspace.remote()
        self.perceptor = Perceptor.remote(GEN_TOKENIZER, GEN_MODEL)
        self.multimodal_perceptor = MultimodalPerceptor()
        self.attention = AttentionModule()
        self.knowledge_extractor = KnowledgeExtractor.remote(GEN_TOKENIZER, GEN_MODEL)
        self.style_learner = StyleLearner(self.user_id)
        self.working_memory = WorkingMemory()
        self.long_term_memory = LongTermMemory.remote(self.user_id)
        self.knowledge_graph = KnowledgeGraph()
        self.memory_manager = MemoryManager.remote(self.user_id, self.long_term_memory, self.knowledge_graph)
        self.emotion_engine = EmotionEngine(self.user_id)
        self.goal_manager = GoalManager(self.user_id)
        self.self_model = SelfModel(self.user_id)
        self.world_model = WorldModel(self.user_id)
        self.imagination = ImaginationModule.remote(self.user_id, self.long_term_memory, GEN_TOKENIZER, GEN_MODEL, self.world_model)
        self.dmn = DefaultModeNetwork.remote(self.user_id, self.long_term_memory, GEN_TOKENIZER, GEN_MODEL)
        self.planner = PlanningModule.remote(self.user_id, self.long_term_memory, GEN_TOKENIZER, GEN_MODEL)
        self.ethics_monitor = EthicsMonitor(GEN_TOKENIZER, GEN_MODEL, self.knowledge_graph)
        self.response_generator = ResponseGenerator.remote(GEN_TOKENIZER, GEN_MODEL)
        self.coherence = CoherenceModule()
        self.prev_response = None
        self.prev_memory_id = None
        self.interaction_id = 0

    async def init(self):
        try:
            await ray.get(self.long_term_memory.load.remote())
            await self.emotion_engine.load()
            await self.self_model.load()
            await self.ethics_monitor.initialize_rules(self.user_id)
            logger.info("Agent initialized successfully")
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")

    async def tick(self):
        try:
            await ray.get(self.long_term_memory.decay.remote())
            await ray.get(self.memory_manager.consolidate.remote())
            narrative = await ray.get(self.dmn.get_narrative.remote())
            await ray.get(self.imagination.simulate.remote(narrative))
            await ray.get(self.dmn.update_narrative.remote())
            await ray.get(self.workspace.broadcast.remote("tick", {"status": "thinking"}, 0.01))
            logger.info("Agent tick executed")
        except Exception as e:
            logger.error(f"Agent tick failed: {e}")

    async def process_file(self, file_path):
        try:
            if file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                result = await self.multimodal_perceptor.process_image(file_path)
            elif file_path.lower().endswith((".mp3", ".wav")):
                result = await self.multimodal_perceptor.process_audio(file_path)
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                return None
            if result:
                await ray.get(self.long_term_memory.store_multimodal.remote(
                    result["embedding"], result["description"], result["type"], salience=0.6
                ))
                fact = await ray.get(self.knowledge_extractor.extract_from_multimodal.remote(result["description"], result["type"]))
                if fact:
                    await self.knowledge_graph.add_fact(
                        self.user_id, fact["subject"], fact["relation"], fact["object"], confidence=0.6, salience=0.6
                    )
                return result
        except Exception as e:
            logger.error(f"File processing failed: {e}")
            return None

    async def respond(self, query, file_path=None):
        try:
            self.interaction_id += 1
            state = deepcopy(self.working_memory.summarize())
            text = query
            if file_path:
                result = await self.process_file(file_path)
                if result:
                    text += f" [{result['type'].capitalize()}: {result['description']}]"
            
            percept = await ray.get(self.perceptor.analyze.remote(text))
            facts = await ray.get(self.knowledge_extractor.extract.remote(text))
            for fact in facts:
                await self.knowledge_graph.add_fact(
                    self.user_id, fact["subject"], fact["relation"], fact["object"], 
                    confidence=0.6, salience=percept["salience"]
                )

            await self.style_learner.update(text)
            goal = self.goal_manager.resolve_conflicts()
            dorsal_weight = self.attention.compute_dorsal_attention(percept, goal)
            ventral_weight = self.attention.compute_ventral_attention(text)
            salience = (percept["salience"] + ventral_weight) * dorsal_weight
            await ray.get(self.workspace.broadcast.remote("percept", percept, salience))
            await ray.get(self.workspace.apply_attention.remote("percept", dorsal_weight + ventral_weight))

            self.working_memory.add(percept)
            self.emotion_engine.update(percept, self.self_model)
            await ray.get(self.long_term_memory.store_semantic.remote(percept["embedding"], text, percept, salience=salience))
            await ray.get(self.long_term_memory.store_episodic.remote(percept, self.prev_memory_id, salience=salience))
            self.prev_memory_id = len(ray.get(self.long_term_memory.episodic_graph.remote()).nodes) - 1

            intent = percept["intent"]
            plan = (
                await ray.get(self.planner.plan.remote(intent, text))
                if intent != "urgent"
                else ["Acknowledge urgency", "Provide immediate assistance", "Follow up"]
            )
            if self.conflict_monitor.detect_conflict(percept, self.prev_response):
                intent = "comfort"
                plan = ["Acknowledge misunderstanding", "Clarify intent", "Offer support"]

            memories = await ray.get([
                self.long_term_memory.retrieve_semantic.remote(percept["embedding"], k=5),
                self.long_term_memory.retrieve_multimodal.remote(percept.get("text_embedding", percept["embedding"]), k=5)
            ])
            memories = memories[0] + memories[1]

            if DUCKLING.parse(query):
                goal = {"type": "reminder", "text": text}
                self.goal_manager.add("short", goal)
                response = "Reminder set."
                reward = 1.0
            else:
                user_style = await self.style_learner.get()
                context = self.coherence.get_context()
                response = await ray.get(self.response_generator.generate.remote(
                    intent, self.workspace, percept["intents"], self.self_model,
                    self.emotion_engine, user_style, plan, context, memories
                ))
                ethics_result = await self.ethics_monitor.evaluate(response, self.user_id)
                if ethics_result["score"] < 0.6 or ethics_result["violations"]:
                    response = "I can't respond in a way that aligns with my ethical guidelines. How else can I assist?"
                    reward = -1.0
                else:
                    reward = (
                        1.5 if any(emoji in query.lower() for emoji in [":smile:", ":+1:", ":heart:"]) or "thank" in query.lower()
                        else -1.0 if any(emoji in [":", ":", "-", "-"] for emoji in query.lower()) or "sorry" in response.lower()
                        else 0.1
                    )

            next_state = deepcopy(self.working_memory.summarize())
            await self.world_model.update(percept["intent"], intent, response, reward)
            
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute(
                    "INSERT INTO metrics_log (user_id, metric_type, value, timestamp) VALUES (?, ?, ?, ?)",
                    (self.user_id, "reward", reward, datetime.utcnow().isoformat())
                )
                await db.commit()

            if reward < 0:
                self.self_model.reflect("failure")
            elif reward > 0:
                self.self_model.reflect("success")

            await self.emotion_engine.save()
            await self.self_model.save()
            self.coherence.add(query, response)
            self.prev_response = response
            return response
        except Exception as e:
            logger.error(f"Response processing failed: {e}")
            return "An error occurred while processing your request. Please try again."

class BillyGUI(QMainWindow):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent
        self.init_ui()
        self.loop = asyncio.get_event_loop()
        self.loop.create_task(self.start())

    def init_ui(self):
        try:
            self.setWindowTitle("Billy AI")
            self.setGeometry(100, 100, 800, 600)
            self.setStyleSheet("""
                QMainWindow { background-color: #f0f0f0; }
                QTextEdit { border: 1px solid #ccc; border-radius: 5px; padding: 5px; font-family: Roboto, sans-serif; }
                QPushButton { 
                    background-color: #007bff; color: white; border: none; 
                    padding: 8px 16px; border-radius: 5px; font-family: Roboto, sans-serif; 
                }
                QPushButton:hover { background-color: #0056b3; }
                QComboBox { border: 1px solid #ccc; border-radius: 5px; padding: 5px; }
                QLabel { font-family: Roboto, sans-serif; }
            """)

            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)

            self.tabs = QTabWidget()
            layout.addWidget(self.tabs)

            chat_widget = QWidget()
            chat_layout = QVBoxLayout(chat_widget)
            
            self.chat_display = QTextEdit()
            self.chat_display.setReadOnly(True)
            chat_layout.addWidget(self.chat_display)

            input_layout = QHBoxLayout()
            self.input_field = QTextEdit()
            self.input_field.setMaximumHeight(50)
            input_layout.addWidget(self.input_field)

            self.send_button = QPushButton("Send")
            self.send_button.clicked.connect(self.send_message)
            input_layout.addWidget(self.send_button)
            chat_layout.addLayout(input_layout)

            file_layout = QHBoxLayout()
            self.upload_button = QPushButton("Upload File")
            self.upload_button.clicked.connect(self.upload_file)
            file_layout.addWidget(self.upload_button)
            self.file_label = QLabel("No file selected")
            file_layout.addWidget(self.file_label)
            chat_layout.addLayout(file_layout)

            self.tabs.addTab(chat_widget, "Chat")

            self.memory_view = QWebEngineView()
            self.tabs.addTab(self.memory_view, "Memory Graph")
            self.knowledge_view = QWebEngineView()
            self.tabs.addTab(self.knowledge_view, "Knowledge Graph")

            settings_widget = QWidget()
            settings_layout = QVBoxLayout(settings_widget)

            style_layout = QHBoxLayout()
            style_label = QLabel("Response Style:")
            style_layout.addWidget(style_label)
            self.style_combo = QComboBox()
            self.style_combo.addItems(["Concise", "Elaborate", "Humorous", "Formal"])
            self.style_combo.currentTextChanged.connect(self.update_style)
            style_layout.addWidget(self.style_combo)
            settings_layout.addLayout(style_layout)

            mood_layout = QHBoxLayout()
            mood_label = QLabel("Mood Sensitivity:")
            mood_layout.addWidget(mood_label)
            self.mood_combo = QComboBox()
            self.mood_combo.addItems(["Low", "Medium", "High"])
            self.mood_combo.setCurrentText("Medium")
            self.mood_combo.currentTextChanged.connect(self.update_mood_sensitivity)
            mood_layout.addWidget(self.mood_combo)
            settings_layout.addLayout(mood_layout)

            viz_layout = QHBoxLayout()
            viz_label = QLabel("Visualization Update Frequency:")
            viz_layout.addWidget(viz_label)
            self.viz_combo = QComboBox()
            self.viz_combo.addItems(["Every Interaction", "Every 5 Interactions", "Manual"])
            self.viz_combo.currentTextChanged.connect(self.update_viz_frequency)
            viz_layout.addWidget(self.viz_combo)
            settings_layout.addLayout(viz_layout)

            self.update_viz_button = QPushButton("Update Visualizations")
            self.update_viz_button.clicked.connect(self.update_visualizations)
            settings_layout.addWidget(self.update_viz_button)

            settings_layout.addStretch()

            self.tabs.addTab(settings_widget, "Settings")

        except Exception as e:
            logger.error(f"UI initialization failed: {e}")
            self.close()

    def update_style(self, style):
        """Update the agent's response style based on user selection."""
        try:
            style_map = {
                "Concise": 0.5,
                "Elaborate": 1.0,
                "Humorous": 0.8,
                "Formal": 0.7
            }
            self.agent.style_learner.style_emb *= style_map.get(style, 0.5)
            logger.info(f"Updated response style to {style}")
        except Exception as e:
            logger.error(f"Style update failed: {e}")

    def update_mood_sensitivity(self, sensitivity):
        """Update the agent's mood sensitivity."""
        try:
            sensitivity_map = {
                "Low": 0.2,
                "Medium": 0.5,
                "High": 0.8
            }
            self.agent.emotion_engine.mood["arousal"] *= sensitivity_map.get(sensitivity, 0.5)
            logger.info(f"Updated mood sensitivity to {sensitivity}")
        except Exception as e:
            logger.error(f"Mood sensitivity update failed: {e}")

    def update_viz_frequency(self, frequency):
        """Update visualization update frequency."""
        try:
            logger.info(f"Visualization update frequency set to {frequency}")
            self.update_viz_button.setEnabled(frequency == "Manual")
        except Exception as e:
            logger.error(f"Visualization frequency update failed: {e}")

    async def update_visualizations(self):
        """Update memory and knowledge graph visualizations."""
        try:
            memory_html = visualize_memory(self.agent.user_id)
            knowledge_html = visualize_knowledge(self.agent.user_id)
            if memory_html:
                self.memory_view.setHtml(memory_html)
            if knowledge_html:
                self.knowledge_view.setHtml(knowledge_html)
            logger.info("Visualizations updated")
        except Exception as e:
            logger.error(f"Visualization update failed: {e}")

    async def send_message(self):
        """Handle sending a message to the agent."""
        try:
            query = self.input_field.toPlainText().strip()
            if not query:
                return
            self.chat_display.append(f"<b>You:</b> {query}")
            self.input_field.clear()
            
            file_path = self.file_path if hasattr(self, 'file_path') else None
            response = await self.agent.respond(query, file_path)
            self.chat_display.append(f"<b>Billy:</b> {response}")

            if self.viz_combo.currentText() == "Every Interaction":
                await self.update_visualizations()

            if file_path:
                self.file_label.setText("No file selected")
                del self.file_path

        except Exception as e:
            logger.error(f"Message sending failed: {e}")
            self.chat_display.append("Error processing your message.")

    def send_message(self):
        """Wrapper to run async send_message in event loop."""
        self.loop.create_task(self.send_message())

    def upload_file(self):
        """Handle file upload."""
        try:
            file_dialog = QFileDialog()
            file_path, _ = file_dialog.getOpenFileName(
                self, "Select File", "",
                "Images (*.png *.jpg *.jpeg);;Audio Files (*.mp3 *.wav)"
            )
            if file_path:
                self.file_path = file_path
                self.file_label.setText(os.path.basename(file_path))
                logger.info(f"File selected: {file_path}")
        except Exception as e:
            logger.error(f"File upload failed: {e}")

    async def start(self):
        """Start the agent's tick loop and initial visualizations."""
        try:
            await self.agent.init()
            await self.update_visualizations()
            while True:
                await self.agent.tick()
                await asyncio.sleep(TICK_INTERVAL)
        except Exception as e:
            logger.error(f"Agent start failed: {e}")
            self.close()

def main():
    """Main function to run the application."""
    try:
        app = QApplication(sys.argv)
        app.setFont(QFont("Roboto", 10))
        agent = CognitiveAgent(USER_ID)
        gui = BillyGUI(agent)
        gui.show()
        sys.exit(app.exec())
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
