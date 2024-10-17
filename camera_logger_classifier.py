import cv2
import os

def take_and_save_picture(capture,label,label2, counter):
    ret, frame = capture.read()
    filename = f"{label}_{counter}.png"
    cv2.imwrite(filename, frame)
    with open('labels2024.txt', 'a') as f:
        f.write(f"{filename}, {label},{label2}\n")
    print(f"Image {filename} saved with label {label}")

def main():
    capture = cv2.VideoCapture(0)

    if not capture.isOpened():
        print("Cannot open camera")
        exit()

    counter = 0
    label = input("Enter initial label: ")
    label2 = input("Enter initial label2: ")
    while True:
        print("Press 'y' to take a picture, 'n' for new label, 'e' to exit: ", end="")
        key = input()

        if key == 'y':
            take_and_save_picture(capture, label,label2, counter)
            counter += 1
        elif key == 'n':
            label = input("Enter new label: ")
            counter = 0
        elif key == 'e':
            break
        else:
            print("Invalid input")

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
