import os
import json

def generate_datasets():
    print("Generating training datasets...")
    os.makedirs("data", exist_ok=True)

    datasets = {
        "react": [
            {
                "instruction": "Create a counter component with increment and decrement buttons",
                "output": "```tsx\nimport { useState } from 'react';\n\nexport const Counter = () => {\n  const [count, setCount] = useState(0);\n  return (\n    <div className=\"flex gap-4 p-4 border rounded\">\n      <button onClick={() => setCount(count - 1)} className=\"px-4 py-2 bg-red-500 text-white rounded\">-</button>\n      <span className=\"text-xl font-bold\">{count}</span>\n      <button onClick={() => setCount(count + 1)} className=\"px-4 py-2 bg-green-500 text-white rounded\">+</button>\n    </div>\n  );\n};\n```"
            },
            {
                "instruction": "Create a product card component with image, name, price and add-to-cart button",
                "output": "```tsx\nimport { useState } from 'react';\n\ninterface ProductProps {\n  id: string;\n  name: string;\n  price: number;\n  image: string;\n}\n\nexport const ProductCard = ({ id, name, price, image }: ProductProps) => {\n  const [added, setAdded] = useState(false);\n  return (\n    <div className=\"border rounded-lg p-4 max-w-sm flex flex-col gap-2\">\n      <img src={image} alt={name} className=\"w-full h-48 object-cover rounded\" />\n      <h3 className=\"font-semibold text-lg\">{name}</h3>\n      <p className=\"text-gray-600\">${price}</p>\n      <button onClick={() => setAdded(true)} className=\"mt-auto bg-blue-600 text-white py-2 rounded\">\n        {added ? 'Added!' : 'Add to Cart'}\n      </button>\n    </div>\n  );\n};\n```"
            },
            {
                "instruction": "Build a user login form with username, password fields and validation in React",
                "output": "```tsx\nimport React, { useState } from 'react';\n\nexport const LoginForm = () => {\n  const [email, setEmail] = useState('');\n  const [password, setPassword] = useState('');\n  \n  const handleSubmit = (e: React.FormEvent) => {\n    e.preventDefault();\n    console.log({ email, password });\n  };\n\n  return (\n    <form onSubmit={handleSubmit} className=\"flex flex-col gap-3 p-6 border rounded-lg\">\n      <input type=\"email\" placeholder=\"Email\" value={email} onChange={e => setEmail(e.target.value)} className=\"border p-2 rounded\" />\n      <input type=\"password\" placeholder=\"Password\" value={password} onChange={e => setPassword(e.target.value)} className=\"border p-2 rounded\" />\n      <button type=\"submit\" className=\"bg-blue-600 text-white py-2 rounded\">Login</button>\n    </form>\n  );\n};\n```"
            }
        ],
        "react_native": [
            {
                "instruction": "Write a React Native counter view",
                "output": "```tsx\nimport React, { useState } from 'react';\nimport { View, Text, TouchableOpacity, StyleSheet } from 'react-native';\n\nexport const RNCounter = () => {\n  const [count, setCount] = useState(0);\n  return (\n    <View style={styles.container}>\n      <TouchableOpacity onPress={() => setCount(count - 1)} style={styles.btn}>\n        <Text style={styles.btnTxt}>-</Text>\n      </TouchableOpacity>\n      <Text style={styles.txt}>{count}</Text>\n      <TouchableOpacity onPress={() => setCount(count + 1)} style={styles.btn}>\n        <Text style={styles.btnTxt}>+</Text>\n      </TouchableOpacity>\n    </View>\n  );\n};\nconst styles = StyleSheet.create({\n  container: { flexDirection: 'row', padding: 20, alignItems: 'center' },\n  btn: { padding: 15, backgroundColor: '#007AFF', borderRadius: 8 },\n  btnTxt: { color: 'white', fontSize: 18 },\n  txt: { marginHorizontal: 20, fontSize: 24, fontWeight: 'bold' }\n});\n```"
            },
            {
                "instruction": "Build a user authentication screen in React Native with JWT login",
                "output": "```tsx\nimport React, { useState } from 'react';\nimport { View, TextInput, Button, StyleSheet, Text } from 'react-native';\n\nexport const LoginScreen = () => {\n  const [username, setUsername] = useState('');\n  const [password, setPassword] = useState('');\n  \n  return (\n    <View style={styles.container}>\n      <Text style={styles.title}>Mobile Login</Text>\n      <TextInput style={styles.input} placeholder=\"Username\" value={username} onChangeText={setUsername} />\n      <TextInput style={styles.input} placeholder=\"Password\" secureTextEntry value={password} onChangeText={setPassword} />\n      <Button title=\"Login\" onPress={() => {}} />\n    </View>\n  );\n};\nconst styles = StyleSheet.create({\n  container: { flex: 1, padding: 20, justifyContent: 'center' },\n  title: { fontSize: 24, marginBottom: 20, textAlign: 'center' },\n  input: { borderHeight: 1, borderColor: '#ccc', padding: 10, marginBottom: 15, borderRadius: 5 }\n});\n```"
            }
        ],
        "python_backend": [
            {
                "instruction": "Write a Django serializer for a Product model with category and price",
                "output": "```python\nfrom rest_framework import serializers\nfrom .models import Product\n\nclass ProductSerializer(serializers.ModelSerializer):\n    class Meta:\n        model = Product\n        fields = ['id', 'name', 'price', 'category', 'created_at']\n        \n    def validate_price(self, value):\n        if value <= 0:\n            raise serializers.ValidationError(\"Price must be positive.\")\n        return value\n```"
            },
            {
                "instruction": "Write the Django REST Framework backend endpoint for login with token generation",
                "output": "```python\nfrom django.contrib.auth import authenticate\nfrom rest_framework.views import APIView\nfrom rest_framework.response import Response\nfrom rest_framework import status\nfrom rest_framework_simplejwt.tokens import RefreshToken\n\nclass LoginAPIView(APIView):\n    permission_classes = []\n    \n    def post(self, request):\n        username = request.data.get('username')\n        password = request.data.get('password')\n        user = authenticate(username=username, password=password)\n        if user:\n            refresh = RefreshToken.for_user(user)\n            return Response({\n                'access': str(refresh.access_token),\n                'refresh': str(refresh),\n                'user_id': user.id\n            }, status=status.HTTP_200_OK)\n        return Response({'detail': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)\n```"
            }
        ],
        "ai_ml": [
            {
                "instruction": "Write a simple PyTorch neural network training step",
                "output": "```python\nimport torch\nimport torch.nn as nn\nimport torch.optim as optim\n\n# Define simple MLP model\nmodel = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 1))\noptimizer = optim.AdamW(model.parameters(), lr=1e-3)\ncriterion = nn.MSELoss()\n\n# Training iteration\nx = torch.randn(8, 10)\ny = torch.randn(8, 1)\n\noptimizer.zero_grad()\npred = model(x)\nloss = criterion(pred, y)\nloss.backward()\noptimizer.step()\nprint(f\"Step complete, Loss: {loss.item():.4f}\")\n```"
            }
        ],
        "merger": [
            {
                "instruction": "Django serializer has user_id (snake_case) but React uses userId (camelCase). Fix the API client.",
                "output": "```typescript\nimport axios from 'axios';\n\ninterface UserData {\n  userId: string;\n  username: string;\n}\n\nexport const fetchUser = async (id: string): Promise<UserData> => {\n  const response = await axios.get(`/api/users/${id}/`);\n  return {\n    userId: response.data.user_id, // aligned snake_case to camelCase\n    username: response.data.username\n  };\n};\n```"
            },
            {
                "instruction": "Integrate React Native user login and Django API JWT auth view outputs",
                "output": "```typescript\n// Integrated Mobile Auth Layer\nimport axios from 'axios';\n\nconst API = axios.create({\n  baseURL: 'http://localhost:8080/api',\n  headers: { 'Content-Type': 'application/json' }\n});\n\nexport const login = async (username, password) => {\n  const response = await API.post('/login/', { username, password });\n  // Maps the backend token response format\n  const { access, refresh, user_id } = response.data;\n  return {\n    token: access,\n    refreshToken: refresh,\n    userId: user_id\n  };\n};\n```"
            }
        ]
    }

    # Write each dataset to corresponding jsonl file
    for domain, items in datasets.items():
        # Ensure we have at least 5 examples by duplicating if necessary, to help training loop
        # We will duplicate the items to have 6 items per file
        items_to_save = items * (6 // len(items) + 1)
        items_to_save = items_to_save[:6]
        
        file_path = f"data/{domain}_corpus.jsonl"
        if domain == "react_native":
            file_path = "data/rn_corpus.jsonl"
        elif domain == "python_backend":
            file_path = "data/python_corpus.jsonl"
            
        with open(file_path, "w", encoding="utf-8") as f:
            for item in items_to_save:
                f.write(json.dumps(item) + "\n")
        print(f"Saved {len(items_to_save)} items to -> {file_path}")

if __name__ == "__main__":
    generate_datasets()
