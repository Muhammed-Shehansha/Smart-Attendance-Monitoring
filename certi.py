import os
import cv2
import csv

list_of_names = []


def delete_old_data():
    for i in os.listdir("generated-certificates/"):
        os.remove("generated-certificates/{}".format(i))


def cleanup_data():
    with open('attendance.csv', newline='') as f:
        ereader = csv.DictReader(f)
        for row in ereader:
            list_of_names.append(row['Name'].strip())


def generate_certificates():
    for index, name in enumerate(list_of_names):
        certificate_template_image = cv2.imread("Template/Certificate of Participation Portrait.jpg")
        name = name.title()
        cv2.putText(certificate_template_image, name.center(24), (105, 453), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.imwrite("generated-certificates/{}.jpg".format(name.strip()), certificate_template_image)
        print("Processing {} / {}".format(index + 1, len(list_of_names)))


def main():
    delete_old_data()
    cleanup_data()
    generate_certificates()


if __name__ == '__main__':
    main()