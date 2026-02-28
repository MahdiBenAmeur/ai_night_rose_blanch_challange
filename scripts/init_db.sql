CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS embeddings (
    id BIGSERIAL PRIMARY KEY,
    id_document INTEGER NOT NULL,
    texte_fragment TEXT NOT NULL,
    vecteur VECTOR(384) NOT NULL
);

-- ivfflat keeps the setup simple and is broadly available in pgvector.
-- vector_cosine_ops enforces cosine-distance search semantics.
CREATE INDEX IF NOT EXISTS embeddings_vecteur_cosine_ivfflat_idx
ON embeddings
USING ivfflat (vecteur vector_cosine_ops)
WITH (lists = 100);
