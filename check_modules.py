from pymongo import MongoClient

# Connect to MongoDB Atlas
client = MongoClient("mongodb+srv://KENSOLOAI:Jecintamugure123@cluster0.lkkaqnv.mongodb.net/?appName=Cluster0")

# Create database and collection
db = client["kensolo_ai"]
collection = db["predictions"]