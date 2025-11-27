import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import numpy as np
import builtins

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestEmbeddingByQwen(unittest.TestCase):

    def test_embedding_flow(self):
        # Read the script content BEFORE any patching of builtins.open
        script_path = os.path.join(os.path.dirname(__file__), '..', 'indexing', 'embedding_by_Qwen.py')
        original_open = builtins.open
        try:
            with original_open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
        finally:
            builtins.open = original_open

        # 1. Prepare Mocks for Modules
        mock_st_module = MagicMock()
        mock_model_cls = MagicMock()
        mock_model_instance = mock_model_cls.return_value
        mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]], dtype='float32')
        mock_model_instance.get_sentence_embedding_dimension.return_value = 1024 # Qwen model dim
        mock_st_module.SentenceTransformer = mock_model_cls

        mock_qc_module = MagicMock()
        mock_client_cls = MagicMock()
        mock_client_instance = mock_client_cls.return_value
        mock_qc_module.QdrantClient = mock_client_cls
        
        mock_models = MagicMock()
        mock_models.VectorParams = MagicMock()
        mock_models.Distance.COSINE = "COSINE" # Qwen uses COSINE distance
        mock_models.HnswConfigDiff = MagicMock()
        mock_models.PointStruct = MagicMock()
        
        mock_qc_module.http.models = mock_models

        mock_dotenv = MagicMock()

        # Mock torch and its cuda functionality
        mock_torch = MagicMock()
        mock_torch.cuda.is_available = MagicMock(return_value=False) # Force CPU path
        mock_torch.cuda.current_device = MagicMock(return_value=0) # Mock current device if needed
        mock_torch.device = MagicMock(return_value="cpu") # Mock torch.device constructor

        # Mock os module carefully, ensure essential functions are passed through
        mock_os = MagicMock(spec=os)
        mock_os.path = MagicMock(spec=os.path)
        mock_os.path.join.side_effect = os.path.join
        mock_os.getenv.side_effect = lambda key, default=None: os.environ.get(key, default)

        # 2. Patch sys.modules and other dependencies
        modules_to_patch = {
            'sentence_transformers': mock_st_module,
            'qdrant_client': mock_qc_module,
            'qdrant_client.http': MagicMock(),
            'qdrant_client.http.models': mock_models,
            'dotenv': mock_dotenv,
            'torch': mock_torch,
            'numpy': np,
            'os': mock_os,
        }

        with patch.dict(sys.modules, modules_to_patch):
            # Patch builtins.open specifically for the data file reading within the script
            mock_data_file_open = mock_open(read_data='{"article_id": "1", "clause_id": "1", "content": "qwen content"}\n')
            with patch('builtins.open', mock_data_file_open):
                with patch.dict(os.environ, {"QDRANT_URL": "mock_url", "QDRANT_API_KEY": "mock_key", "CUDA_VISIBLE_DEVICES": ""}):
                    
                    # 3. Execute Script
                    global_vars = {
                        '__file__': script_path,
                        '__name__': '__main__',
                    }
                    exec(script_content, global_vars)

                    # 4. Assertions
                    mock_data_file_open.assert_called_with("data/Retrieval/semantic_chunking_for_embedding.jsonl", "r", encoding="utf-8")
                    
                    # Qwen specific: model name
                    mock_model_cls.assert_called_with(
                        "models/Qwen3-Embedding-0.6B"
                    )
                    
                    mock_model_instance.encode.assert_called()
                    call_kwargs = mock_model_instance.encode.call_args.kwargs
                    self.assertEqual(call_kwargs.get('prompt_name'), "document") # Qwen specific encode param
                    self.assertEqual(call_kwargs.get('batch_size'), 8) # Qwen specific batch size
                    
                    mock_client_cls.assert_called_with(url="mock_url", api_key="mock_key")
                    
                    # Qwen script does not call delete_collection
                    mock_client_instance.create_collection.assert_called()
                    self.assertEqual(
                        mock_client_instance.create_collection.call_args.kwargs['collection_name'], 
                        "legal_clauses_Qwen3"
                    )
                    # Assert Distance.COSINE (checked by mocking mock_models.Distance.COSINE explicitly)
                    
                    mock_client_instance.upsert.assert_called()

if __name__ == '__main__':
    unittest.main()
