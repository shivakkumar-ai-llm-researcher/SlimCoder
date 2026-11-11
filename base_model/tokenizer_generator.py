import os
from tokenizers import ByteLevelBPETokenizer

def generate_tokenizer():
    print("Generating BPE Tokenizer...")
    # Create target directories
    os.makedirs("base_model/tokenizer", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Create a small dummy corpus to bootstrap tokenizer vocabulary
    corpus_text = """
    import os
    import sys
    import torch
    import torch.nn as nn
    from typing import Dict, List, Optional
    
    # React Component example
    import React, { useState, useEffect } from 'react';
    export const Button = ({ label, onClick }) => {
        const [clicked, setClicked] = useState(false);
        return <button onClick={onClick}>{label}</button>;
    };

    # React Native Screen
    import { StyleSheet, Text, View, TouchableOpacity } from 'react-native';
    export default function App() {
        return (
            <View style={styles.container}>
                <Text>React Native SLM System</Text>
            </View>
        );
    }

    # Django API View
    from rest_framework import viewsets, permissions
    from rest_framework.response import Response
    class UserViewSet(viewsets.ModelViewSet):
        permission_classes = [permissions.IsAuthenticated]
        def list(self, request):
            return Response({"status": "ok"})

    # AI/ML PyTorch
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)
        def forward(self, x):
            return self.linear(x)

    # Standard conversation corpus
    You are an expert developer. Design a clean system.
    Please write the code for user login component in React.
    Explain the difference between a language model and a LoRA adapter.
    Hot-swap the LoRA weights at runtime.
    Align Django API serializers with typescript frontend models.
    """
    
    corpus_path = "base_model/tokenizer/corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write(corpus_text * 10) # Duplicate to ensure sufficient frequency

    # Initialize and train ByteLevelBPETokenizer
    tokenizer = ByteLevelBPETokenizer()
    
    # We define special tokens. Hugging Face tokenizers will map these to specific IDs (typically 0, 1, 2, 3...)
    special_tokens = [
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|end|>",
        "<|pad|>"
    ]
    
    tokenizer.train(
        files=[corpus_path],
        vocab_size=32000,
        min_frequency=1,
        special_tokens=special_tokens
    )

    # Save tokenizer to the expected directory
    tokenizer.save("base_model/tokenizer/shiv_tokenizer.json")
    print("Tokenizer saved successfully at: base_model/tokenizer/shiv_tokenizer.json")
    
    # Clean up corpus file
    if os.path.exists(corpus_path):
        os.remove(corpus_path)

if __name__ == "__main__":
    generate_tokenizer()
