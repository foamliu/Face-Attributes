from utils_aip import get_face_attributes

if __name__ == "__main__":
    filePath = "images/003ac.jpg"

    attr = get_face_attributes(filePath)

    print(attr)
