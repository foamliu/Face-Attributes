import base64

from aip.face import AipFace

appId = '16217748'
apiKey = 'fq8w67yHRje2Tmr1UKhD9lCs'
secretKey = 'srTGstYbXL4cDmbGM8iVF7amyqFbxPfN'

aipFace = AipFace(appId, apiKey, secretKey)


def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()


options = {}
options["face_field"] = "age,beauty,expression,faceshape,gender,glasses,race,quality,facetype"
options["max_face_num"] = 1
options["face_type"] = "LIVE"

filePath = "images/003ac.jpg"
image = base64.b64encode(get_file_content(filePath)).decode()
result = aipFace.detect(image=image, image_type="BASE64", options=options)
print(result)
