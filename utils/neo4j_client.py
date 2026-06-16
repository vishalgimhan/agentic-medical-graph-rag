from typing import List, Optional, Dict, Any

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

    # -- Context Manager --

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

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

    def batch_write(self, cypher: str, param_list: List[Dict[str, Any]]) -> int:
        """Execute a write for each param dict in the list using UNWIND.

        The cypher should reference `$batch` as an UNWIND variable.
        Example:
            UNWIND $batch AS row
            MERGE (d:Drug {name: row.name})
            SET d.description = row.description

        Returns the number of items processed.
        """
        if not param_list:
            return 0
        with self._driver.session(database=self._database) as session:
            session.run(cypher, batch=param_list)
        return len(param_list)

    # -- query --
    def query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute a read query and return list of record dicts."""
        params = params or {}
        with self._driver.session(database=self._database) as session:
            result = session.run(cypher, **params)
            return [dict(record) for record in result]
        
    # -- Introspection --

    def get_node_count(self, label: Optional[str] = None) -> int:
        """Count nodes, optionally filtered by label."""
        if label:
            result = self.query(f"MATCH (n:`{label}`) RETURN count(n) AS cnt")
        else:
            result = self.query("MATCH (n) RETURN count(n) AS cnt")
        return result[0]["cnt"] if result else 0

    def get_relationship_count(self, rel_type: Optional[str] = None) -> int:
        """Count relationships, optionally filtered by type."""
        if rel_type:
            result = self.query(f"MATCH ()-[r:`{rel_type}`]->() RETURN count(r) AS cnt")
        else:
            result = self.query("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return result[0]["cnt"] if result else 0

    def get_schema_info(self) -> Dict[str, Any]:
        """Introspect the graph for labels, relationship types, and counts."""
        labels = self.query("CALL db.labels() YIELD label RETURN label")
        rel_types = self.query("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")

        info = {
            "labels": {},
            "relationship_types": {},
            "total_nodes": self.get_node_count(),
            "total_relationships": self.get_relationship_count(),
        }

        for row in labels:
            lbl = row["label"]
            info["labels"][lbl] = self.get_node_count(lbl)

        for row in rel_types:
            rt = row["relationshipType"]
            info["relationship_types"][rt] = self.get_relationship_count(rt)

        return info
    
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