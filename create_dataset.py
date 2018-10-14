import cv2
import glob

VIDEO_PATH = './video/'
OUTPUT_PATH = './model_another/'
OUT_FACE_PATH = './dataset/'
XML_PATH = "./lbpcascade_animeface.xml"


def movie_to_image(num_cut):

    videos = glob.glob(VIDEO_PATH + "*.mp4")
    img_count = 0
    frame_count = 0
    for video in videos:
        capture = cv2.VideoCapture(video)
        while(capture.isOpened()):
            ret, frame = capture.read()
            if ret == False:
                break
            if frame_count % num_cut == 0:
                img_file_name = OUTPUT_PATH + str(img_count) + ".jpg"
                cv2.imwrite(img_file_name, frame)
                img_count += 1

            frame_count += 1

        capture.release()


def face_detect(img_list):

    classifier = cv2.CascadeClassifier(XML_PATH)

    img_count = 127000
    for img_path in img_list:

        org_img = cv2.imread(img_path, cv2.IMREAD_COLOR)

        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)


        
        face_points = classifier.detectMultiScale(gray_img,
                                                  scaleFactor=1.2, minNeighbors=2, minSize=(1, 1))

        for points in face_points:

            x, y, width, height = points

            dst_img = org_img[100:y+height, x-5:x+width] #########

            face_img = cv2.resize(dst_img, (96, 96))
            new_img_name = OUT_FACE_PATH + str(img_count) + 'face.jpg'
            cv2.imwrite(new_img_name, face_img)
            img_count += 1


if __name__ == '__main__':

    #movie_to_image(int(10))

    images = glob.glob(OUTPUT_PATH + '*.jpg')
    face_detect(images)
