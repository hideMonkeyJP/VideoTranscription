import requests
from PIL import Image
import io

class GyazoClient:
    def __init__(self, access_token):
        """GyazoClientの初期化"""
        self.access_token = access_token
        self.upload_url = "https://upload.gyazo.com/api/upload"
        
    def _pil_to_binary(self, pil_img, format="PNG"):
        """PIL画像をバイナリデータに変換"""
        if pil_img.mode == "RGBA" and format == "JPEG":
            pil_img = pil_img.convert("RGB")
        buffered = io.BytesIO()
        pil_img.save(buffered, format=format)
        return buffered.getvalue()
    
    def upload_image(self, image_path, description=""):
        """画像をアップロードしてURLを取得"""
        try:
            # 画像を開く
            img = Image.open(image_path)
            
            # ヘッダーとファイルの準備
            headers = {'Authorization': f"Bearer {self.access_token}"}
            img_binary = self._pil_to_binary(img)
            files = {'imagedata': ('image.png', img_binary, 'image/png')}
            
            # アップロードリクエスト
            response = requests.post(
                self.upload_url,
                headers=headers,
                files=files,
                data={"desc": description}
            )
            response.raise_for_status()
            
            # レスポンスからURLを取得
            response_data = response.json()
            if "url" in response_data:
                return response_data["url"]
            else:
                print("URLが見つかりませんでした")
                return None
                
        except Exception as e:
            print(f"画像のアップロードに失敗: {str(e)}")
            return None