"""
Protein Language Models (PLMs) for sequence analysis and representation learning.

This module provides interfaces to various protein language models including
ESM-2, ProtBERT, and AlphaFold embeddings for protein sequence analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from loguru import logger
import esm


class ProteinLanguageModel:
    """
    Unified interface for protein language models.
    
    Supports multiple PLM architectures:
    - ESM-2 (Facebook AI Research)
    - ProtBERT (Rostlab)
    - Custom fine-tuned models
    """
    
    def __init__(
        self,
        model_name: str = "facebook/esm2_t33_650M_UR50D",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize protein language model.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to load model on ('cuda', 'cpu', or None for auto)
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        logger.info(f"Loading protein language model: {model_name}")
        self._load_model()
        
    def _load_model(self):
        """Load the specified protein language model."""
        try:
            if "esm" in self.model_name.lower():
                self._load_esm_model()
            else:
                self._load_huggingface_model()
                
            logger.success(f"Successfully loaded {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name}: {e}")
            raise
            
    def _load_esm_model(self):
        """Load ESM model from fair-esm library."""
        if "esm2_t33_650M" in self.model_name:
            self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        elif "esm2_t30_150M" in self.model_name:
            self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        else:
            # Fallback to HuggingFace
            self._load_huggingface_model()
            return
            
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def _load_huggingface_model(self):
        """Load model from HuggingFace transformers."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_dir
        )
        self.model = AutoModel.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir
        ).to(self.device)
        self.model.eval()
        
    def encode_sequences(
        self, 
        sequences: Union[str, List[str]], 
        return_attention: bool = False,
        layer: int = -1
    ) -> Dict[str, torch.Tensor]:
        """
        Encode protein sequences into embeddings.
        
        Args:
            sequences: Single sequence or list of sequences
            return_attention: Whether to return attention weights
            layer: Which layer to extract representations from (-1 for last)
            
        Returns:
            Dictionary containing embeddings and optionally attention weights
        """
        if isinstance(sequences, str):
            sequences = [sequences]
            
        with torch.no_grad():
            if hasattr(self, 'alphabet'):  # ESM model
                return self._encode_esm(sequences, return_attention, layer)
            else:  # HuggingFace model
                return self._encode_huggingface(sequences, return_attention, layer)
                
    def _encode_esm(
        self, 
        sequences: List[str], 
        return_attention: bool, 
        layer: int
    ) -> Dict[str, torch.Tensor]:
        """Encode sequences using ESM model."""
        # Prepare data
        data = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        
        # Forward pass
        results = self.model(
            batch_tokens, 
            repr_layers=[layer] if layer != -1 else [self.model.num_layers],
            return_contacts=return_attention
        )
        
        # Extract representations
        layer_key = layer if layer != -1 else self.model.num_layers
        embeddings = results["representations"][layer_key]
        
        output = {
            "embeddings": embeddings,
            "tokens": batch_tokens,
            "sequence_lengths": [len(seq) for seq in sequences]
        }
        
        if return_attention:
            output["attention"] = results.get("contacts")
            
        return output
        
    def _encode_huggingface(
        self, 
        sequences: List[str], 
        return_attention: bool, 
        layer: int
    ) -> Dict[str, torch.Tensor]:
        """Encode sequences using HuggingFace model."""
        # Tokenize sequences
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=1024
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Forward pass
        outputs = self.model(
            **encoded,
            output_hidden_states=True,
            output_attentions=return_attention
        )
        
        # Extract embeddings from specified layer
        if layer == -1:
            embeddings = outputs.last_hidden_state
        else:
            embeddings = outputs.hidden_states[layer]
            
        output = {
            "embeddings": embeddings,
            "attention_mask": encoded["attention_mask"],
            "sequence_lengths": encoded["attention_mask"].sum(dim=1).cpu().tolist()
        }
        
        if return_attention:
            output["attention"] = outputs.attentions
            
        return output
        
    def get_sequence_embeddings(
        self, 
        sequences: Union[str, List[str]],
        pooling: str = "mean"
    ) -> np.ndarray:
        """
        Get pooled sequence-level embeddings.
        
        Args:
            sequences: Protein sequences
            pooling: Pooling strategy ('mean', 'max', 'cls')
            
        Returns:
            Sequence embeddings as numpy array
        """
        encoded = self.encode_sequences(sequences)
        embeddings = encoded["embeddings"]
        
        if pooling == "mean":
            # Average over sequence length (excluding padding)
            if "attention_mask" in encoded:
                mask = encoded["attention_mask"].unsqueeze(-1)
                embeddings = (embeddings * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                # For ESM, manually compute mean excluding special tokens
                embeddings = embeddings[:, 1:-1].mean(dim=1)  # Exclude <cls> and <eos>
        elif pooling == "max":
            embeddings = embeddings.max(dim=1)[0]
        elif pooling == "cls":
            embeddings = embeddings[:, 0]  # CLS token
        else:
            raise ValueError(f"Unsupported pooling strategy: {pooling}")
            
        return embeddings.cpu().numpy()
        
    def get_residue_embeddings(
        self, 
        sequences: Union[str, List[str]]
    ) -> List[np.ndarray]:
        """
        Get per-residue embeddings for each sequence.
        
        Args:
            sequences: Protein sequences
            
        Returns:
            List of per-residue embedding arrays
        """
        encoded = self.encode_sequences(sequences)
        embeddings = encoded["embeddings"]
        seq_lengths = encoded["sequence_lengths"]
        
        residue_embeddings = []
        for i, length in enumerate(seq_lengths):
            if hasattr(self, 'alphabet'):  # ESM model
                # Exclude special tokens
                seq_emb = embeddings[i, 1:length+1].cpu().numpy()
            else:  # HuggingFace model
                # Use attention mask to get actual sequence
                mask = encoded["attention_mask"][i]
                seq_emb = embeddings[i][mask.bool()][1:-1].cpu().numpy()  # Exclude special tokens
                
            residue_embeddings.append(seq_emb)
            
        return residue_embeddings
        
    def compute_similarity(
        self, 
        seq1: str, 
        seq2: str, 
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two protein sequences.
        
        Args:
            seq1, seq2: Protein sequences to compare
            metric: Similarity metric ('cosine', 'euclidean')
            
        Returns:
            Similarity score
        """
        embeddings = self.get_sequence_embeddings([seq1, seq2])
        emb1, emb2 = embeddings[0], embeddings[1]
        
        if metric == "cosine":
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        elif metric == "euclidean":
            similarity = -np.linalg.norm(emb1 - emb2)  # Negative distance
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
            
        return float(similarity)
        
    def analyze_sequence(self, sequence: str) -> Dict[str, any]:
        """
        Comprehensive sequence analysis using the protein language model.
        
        Args:
            sequence: Protein sequence to analyze
            
        Returns:
            Dictionary with various analysis results
        """
        # Get embeddings and attention
        encoded = self.encode_sequences([sequence], return_attention=True)
        seq_embedding = self.get_sequence_embeddings([sequence])[0]
        residue_embeddings = self.get_residue_embeddings([sequence])[0]
        
        analysis = {
            "sequence": sequence,
            "length": len(sequence),
            "sequence_embedding": seq_embedding,
            "residue_embeddings": residue_embeddings,
            "embedding_dim": seq_embedding.shape[0],
        }
        
        # Add attention analysis if available
        if "attention" in encoded and encoded["attention"] is not None:
            attention = encoded["attention"]
            if hasattr(attention, 'shape'):
                analysis["attention_patterns"] = attention.cpu().numpy()
                analysis["attention_shape"] = attention.shape
                
        return analysis
        
    def __repr__(self) -> str:
        return f"ProteinLanguageModel(model={self.model_name}, device={self.device})"