# Конфигурация для LM Studio
from SemanticDeduplicator import SemanticDeduplicator
from configs.llm_configs import lm_studio_config
from db_helper import Neo4jLoader
from services.CacheService import CacheService
from universal_document_analyzer import UniversalMultimodalAnalyzer
from QdrantVectorLoader import QdrantVectorLoader

# Инициализация анализатора
cache = CacheService()
neo4j_loader = Neo4jLoader("neo4j://localhost:7687", "neo4j", "testpass")
qdrant_loader = QdrantVectorLoader(lm_studio_config, cache, "localhost", 6333, "system_artifacts")
semantic_deduplicator = SemanticDeduplicator(lm_studio_config, qdrant_loader, neo4j_loader, cache)
analyzer = UniversalMultimodalAnalyzer(lm_studio_config, qdrant_loader)

results = analyzer.analyze_file(
    file_path="docs/usermanualshort2.docx"
)

for req in results["requirements"]:
    processed = semantic_deduplicator.process_single_requirement(req)

# updated_requirements = semantic_deduplicator.process_single_requirement(results["requirements"][0])
# updated_requirements = semantic_deduplicator.process_requirements(results["requirements"])
# results["requirements"] = updated_requirements
#
# embedded_results = qdrant_loader.build_embeddings(results)
# qdrant_loader.upload_embeddings(embedded_results)
# neo4j_loader.load_to_neo4j(results)
