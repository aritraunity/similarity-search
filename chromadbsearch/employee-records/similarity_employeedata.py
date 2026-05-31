import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name='all-MiniLM-L6-v2')

client = chromadb.Client()

collection_name = "employee_collection"

collection = client.create_collection(
    name=collection_name,
    metadata={
        "description": "Records for Employee Data"
    },
    configuration={
        "embedding_function": ef,
        "hnsw":{
            "space":"cosine"
        }
    }
)
# Defining a list of employee dictionaries
# Each dictionary represents an individual employee with comprehensive information
employees = [
    {
        "id": "employee_1",
        "name": "John Doe",
        "experience": 5,
        "department": "Engineering",
        "role": "Software Engineer",
        "skills": "Python, JavaScript, React, Node.js, databases",
        "location": "New York",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_2",
        "name": "Jane Smith",
        "experience": 8,
        "department": "Marketing",
        "role": "Marketing Manager",
        "skills": "Digital marketing, SEO, content strategy, analytics, social media",
        "location": "Los Angeles",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_3",
        "name": "Alice Johnson",
        "experience": 3,
        "department": "HR",
        "role": "HR Coordinator",
        "skills": "Recruitment, employee relations, HR policies, training programs",
        "location": "Chicago",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_4",
        "name": "Michael Brown",
        "experience": 12,
        "department": "Engineering",
        "role": "Senior Software Engineer",
        "skills": "Java, Spring Boot, microservices, cloud architecture, DevOps",
        "location": "San Francisco",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_5",
        "name": "Emily Wilson",
        "experience": 2,
        "department": "Marketing",
        "role": "Marketing Assistant",
        "skills": "Content creation, email marketing, market research, social media management",
        "location": "Austin",
        "employment_type": "Part-time"
    },
    {
        "id": "employee_6",
        "name": "David Lee",
        "experience": 15,
        "department": "Engineering",
        "role": "Engineering Manager",
        "skills": "Team leadership, project management, software architecture, mentoring",
        "location": "Seattle",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_7",
        "name": "Sarah Clark",
        "experience": 8,
        "department": "HR",
        "role": "HR Manager",
        "skills": "Performance management, compensation planning, policy development, conflict resolution",
        "location": "Boston",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_8",
        "name": "Chris Evans",
        "experience": 20,
        "department": "Engineering",
        "role": "Senior Architect",
        "skills": "System design, distributed systems, cloud platforms, technical strategy",
        "location": "New York",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_9",
        "name": "Jessica Taylor",
        "experience": 4,
        "department": "Marketing",
        "role": "Marketing Specialist",
        "skills": "Brand management, advertising campaigns, customer analytics, creative strategy",
        "location": "Miami",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_10",
        "name": "Alex Rodriguez",
        "experience": 18,
        "department": "Engineering",
        "role": "Lead Software Engineer",
        "skills": "Full-stack development, React, Python, machine learning, data science",
        "location": "Denver",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_11",
        "name": "Hannah White",
        "experience": 6,
        "department": "HR",
        "role": "HR Business Partner",
        "skills": "Strategic HR, organizational development, change management, employee engagement",
        "location": "Portland",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_12",
        "name": "Kevin Martinez",
        "experience": 10,
        "department": "Engineering",
        "role": "DevOps Engineer",
        "skills": "Docker, Kubernetes, AWS, CI/CD pipelines, infrastructure automation",
        "location": "Phoenix",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_13",
        "name": "Rachel Brown",
        "experience": 7,
        "department": "Marketing",
        "role": "Marketing Director",
        "skills": "Strategic marketing, team leadership, budget management, campaign optimization",
        "location": "Atlanta",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_14",
        "name": "Matthew Garcia",
        "experience": 3,
        "department": "Engineering",
        "role": "Junior Software Engineer",
        "skills": "JavaScript, HTML/CSS, basic backend development, learning frameworks",
        "location": "Dallas",
        "employment_type": "Full-time"
    },
    {
        "id": "employee_15",
        "name": "Olivia Moore",
        "experience": 12,
        "department": "Engineering",
        "role": "Principal Engineer",
        "skills": "Technical leadership, system architecture, performance optimization, mentoring",
        "location": "San Francisco",
        "employment_type": "Full-time"
    },
]

# Create comprehensive text documents for each employee
# These documents will be used for similarity search based on skills, roles, and experience
employee_documents = []
for employee in employees:
    document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
    document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
    document += f"Employment type: {employee['employment_type']}."
    employee_documents.append(document)

# Adding data to the collection in the Chroma database
# The 'add' method inserts or updates data into the specified collection
collection.add(
    # Extracting employee IDs to be used as unique identifiers for each record
    ids=[employee["id"] for employee in employees],
    # Using the comprehensive text documents we created
    documents=employee_documents,
    # Adding comprehensive metadata for filtering and search
    metadatas=[{
        "name": employee["name"],
        "department": employee["department"],
        "role": employee["role"],
        "experience": employee["experience"],
        "location": employee["location"],
        "employment_type": employee["employment_type"]
    } for employee in employees]
)

query_text = "Python developer with web development experience"
results = collection.query(
    query_texts=[query_text],
    n_results=3
)

query_text = "team leader manager with experience"
results = collection.query(
    query_texts=[query_text],
    n_results=3
)