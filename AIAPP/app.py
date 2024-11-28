# app.py

from flask import Flask, request, jsonify
import model_handler

app = Flask(__name__)

# 모델 로드 (서버가 시작될 때 한 번만 로드)
model_path = "model.pth"  # 모델 파일 경로
in_features = 14  # 입력 특성의 수
hidden_features = [512, 512, 256, 256]  # 은닉층 특성의 수
out_features = 1 # 출력 특성의 수
model = model_handler.load_model(model_path,in_features,hidden_features,out_features)  # 모델 로드
@app.route('/')
def home():
    return 'Hello, World'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 클라이언트에서 JSON 데이터 받기
        data = request.json
        input_data = data['input']  # 'input' 키로 받은 데이터

        # 모델 예측 수행
        output = model_handler.predict(model, input_data)

        # 예측 결과를 JSON 형식으로 반환
        return jsonify({"output": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
