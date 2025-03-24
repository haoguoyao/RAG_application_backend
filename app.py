import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # 导入 CORS
from build_index import get_chroma_index,build_chroma_index,get_query_engine,check_chroma_index
from parse_document import hash_file_chunked,parse_pdf,parse_html
from keyword_search import save_chunk_text,load_pdf_text,parse_pdf_for_keyword_search,keyword_search,parse_html_for_keyword_search
from flask import Response
# 初始化 Flask 应用
app = Flask(__name__)

# 允许所有来源的跨域请求
CORS(app)

# 定义上传目录
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'pdf', 'html'}

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    print("upload_file request received")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF and HTML files are allowed"}), 400

    # 生成文件路径并保存
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)
    file_ext = os.path.splitext(file.filename)[1].lower()

    document_hash = hash_file_chunked(file_path)
    if check_chroma_index(document_hash):
        return jsonify({"message": "RAG already established"}), 200
    response = upload_file_worker(file_path)



    return response


def upload_file_worker(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    print(f"file_ext: {file_ext}")
    document_hash = hash_file_chunked(file_path)
    if file_ext == '.pdf':
        documents = parse_pdf(file_path, chunk_size=1000, overlap=100)
        
        index = build_chroma_index(documents, document_hash)
        pdf_text = parse_pdf_for_keyword_search(file_path)
        save_chunk_text(pdf_text, f"uploads/{document_hash}.json")
    elif file_ext == '.html':
        print("html file")
        documents = parse_html(file_path, chunk_size=1000, overlap=100)
        index = build_chroma_index(documents, document_hash)
        html_text = parse_html_for_keyword_search(file_path)
        save_chunk_text(html_text, f"uploads/{document_hash}.json")
    else:
        return jsonify({"error": "Unsupported file type"}), 400


    return jsonify({"message": "RAG established"}), 200

@app.route('/search', methods=['POST'])
def search():
    try:
        # Parse request JSON
        data = request.get_json()

        # Extract query parameters
        query = data.get("query", "")
        search_type = data.get("searchType", "")  # Default to "semantic"
        document_hash = data.get("hash", "")  # File hash from frontend

        # Print received request details
        print(f"Received search request: Query='{query}', Search Type='{search_type}', Document Hash='{document_hash}'")
        if search_type == "keyword":
            pdf_text = load_pdf_text(f"uploads/{document_hash}.json")
            streaming_response = keyword_search(pdf_text, query)
        else:
            query_engine = get_query_engine(document_hash)  # Assume a function to retrieve the index
            streaming_response = query_engine.query(query)  # Get the streaming response
            streaming_response = streaming_response.response_gen


        # Define a generator function for streaming
        def generate():
            for chunk in streaming_response:
                yield chunk.encode("utf-8")  # Send each chunk to the client
            yield "\n"  # Send a final newline for proper ending

        # Return a streaming response
        return Response(generate(), content_type="text/plain")

    except Exception as e:
        print(f"Error processing search request: {e}")
        return jsonify({"error": "Internal server error"}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)