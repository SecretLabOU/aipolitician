from pymilvus import connections, Collection, utility
import json

print("Connecting to Milvus database...")
connections.connect(alias="default", host="localhost", port="19530")

# Check available collections
collections = utility.list_collections()
print(f"\nAvailable collections: {collections}")

# Check if our collection exists
collection_name = "political_figures"
exists = utility.has_collection(collection_name)
print(f"\nCollection '{collection_name}' exists: {exists}")

if exists:
    # Get collection details
    collection = Collection(collection_name)
    collection.load()
    
    # Check entity count
    entity_count = collection.num_entities
    print(f"\nEntity count: {entity_count}")
    
    # Display schema structure
    print("\nSchema structure:")
    schema = collection.schema
    schema_dict = {
        "name": collection.name,
        "description": collection.description,
        "primary_field": None,
        "fields": []
    }
    
    # Extract field details
    for field in schema.fields:
        field_info = {
            "name": field.name,
            "type": str(field.dtype).split('.')[-1],
            "is_primary": field.is_primary
        }
        if field.is_primary:
            schema_dict["primary_field"] = field.name
        if hasattr(field, "params") and field.params:
            field_info["params"] = field.params
        schema_dict["fields"].append(field_info)
    
    # Pretty print schema
    print(json.dumps(schema_dict, indent=2))
    
    # Get index information if available
    print("\nIndex information:")
    try:
        index_info = collection.index()
        print(json.dumps(index_info, indent=2))
    except Exception as e:
        print(f"Could not retrieve index information: {e}")
else:
    print(f"Collection '{collection_name}' does not exist!")
