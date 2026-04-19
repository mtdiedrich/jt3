(SELECT 'embeddings.clues' AS table_name, array_length(embedding, 1) AS dimensions FROM embeddings.clues LIMIT 1)
UNION ALL
(SELECT 'embeddings.responses', array_length(embedding, 1) FROM embeddings.responses LIMIT 1)
UNION ALL
(SELECT 'embeddings.categories', array_length(embedding, 1) FROM embeddings.categories LIMIT 1);
