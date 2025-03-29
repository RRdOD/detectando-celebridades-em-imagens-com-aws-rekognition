from pathlib import Path
import boto3
from mypy_boto3_rekognition.type_defs import (
    CelebrityTypeDef,
    RecognizeCelebritiesResponseTypeDef,
)
from PIL import Image, ImageDraw, ImageFont

# Inicializar o cliente do Rekognition
client = boto3.client("rekognition")


def get_path(file_name: str) -> str:
    """Obtém o caminho do arquivo a partir do diretório atual"""
    return str(Path(__file__).parent / "imagens" / file_name)


def recognize_celebrities(photo: str) -> RecognizeCelebritiesResponseTypeDef:
    """Detecta celebridades em uma imagem usando AWS Rekognition"""
    with open(photo, "rb") as image:
        return client.recognize_celebrities(Image={"Bytes": image.read()})


def draw_boxes(image_path: str, output_path: str, face_details: list[CelebrityTypeDef]):
    """Desenha retângulos ao redor das celebridades detectadas e escreve os nomes"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r"C:\Windows\Fonts\Georgia.ttf", 20)

    width, height = image.size

    for face in face_details:
        box = face["Face"]["BoundingBox"]  # type: ignore
        left = int(box["Left"] * width)  # type: ignore
        top = int(box["Top"] * height)  # type: ignore
        right = int((box["Left"] + box["Width"]) * width)  # type: ignore
        bottom = int((box["Top"] + box["Height"]) * height)  # type: ignore

        confidence = face.get("MatchConfidence", 0)
        if confidence > 90:
            # Desenha o retângulo vermelho
            draw.rectangle([left, top, right, bottom], outline="red", width=3)

            # Escreve o nome da celebridade
            text = face.get("Name", "")
            position = (left, top - 20)
            bbox = draw.textbbox(position, text, font=font)  # type: ignore
            draw.rectangle(bbox, fill="red")
            draw.text(position, text, font=font, fill="white")

    image.save(output_path)
    print(f"Imagem salva com resultados em : {output_path}")


def comparar_imagens(imagem1_path: str, imagem2_path: str):
    """Compara duas imagens e retorna se elas têm rostos semelhantes"""
    with open(imagem1_path, 'rb') as img1_file, open(imagem2_path, 'rb') as img2_file:
        img1_bytes = img1_file.read()
        img2_bytes = img2_file.read()

    response = client.compare_faces(
        SourceImage={'Bytes': img1_bytes},
        TargetImage={'Bytes': img2_bytes}
    )

    return response


def main():
    photo_paths = [
        get_path("bbc.jpg"),
        get_path("msn.jpg"),
        get_path("neymar-torcedores.jpg"),
    ]

    for photo_path in photo_paths:
        response = recognize_celebrities(photo_path)
        faces = response["CelebrityFaces"]
        if not faces:
            print(f"Não foram encontrados famosos para a imagem: {photo_path}")
            continue
        output_path = get_path(f"{Path(photo_path).stem}-resultado.jpg")
        draw_boxes(photo_path, output_path, faces)

    # Comparar imagens de exemplo
    imagem1_path = get_path('bbc.jpg')  # Substitua com a sua imagem
    imagem2_path = get_path('msn.jpg')  # Substitua com a sua imagem
    comparacao = comparar_imagens(imagem1_path, imagem2_path)
    print("Resultado da comparação:", comparacao)


if __name__ == "__main__":
    main()
