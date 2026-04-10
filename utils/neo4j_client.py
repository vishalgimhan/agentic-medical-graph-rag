from typing import Optional, Dict, Any

from neo4j import GraphDatabase
from loguru import logger

class Neo4jClient:
    """
    Thread-safe Neo4j driver wrapper with context-manager support.
    """
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        self._uri = uri
        self._database = database
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    # -- connection --
    def verify_connectivity(self) -> bool:
        """
        Verify connectivity to the Neo4j database.
        """
        try:
            self._driver.verify_connectivity()
            logger.success(f"Connected to Neo4j at {self._uri}")
            return True
        except Exception as e:
            logger.error(f"Neo4j connected failed: {e}")
            return False
        
    def close(self):
        """
        Close the Neo4j driver connection.
        """
        self._driver.close()
        logger.info("Neo4j connection closed.")

    # -- write --
    def write(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Execute a single wirte statement
        """
        params = params or {}
        with self._driver.session(database=self._database) as session:
            session.run(cypher, **params)

def get_neo4j_client() -> Neo4jClient:
    """
    Factory: Create a Neo4jClient from environment config"""
    from utils.config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE

    if not NEO4J_URI:
        raise ValueError("NEO4J_URI is not set in environment variables.")

    if not NEO4J_PASSWORD:
        raise ValueError("NEO4J_PASSWORD is not set in environment variables.")
    
    return Neo4jClient(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        database=NEO4J_DATABASE
    )