import cv2 as cv

frame_path = "data/results"

def main():
    first_image = cv.imread(frame_path + "/result_1.png")
    if first_image is None:
        print("cannot get image")
        exit()
    frame_width, frame_height, _ = first_image.shape


    fourcc = cv.VideoWriter_fourcc(*"PNG ")
    out = cv.VideoWriter("output.avi", fourcc, 30.0, (frame_width, frame_height))
    for i in range(644):
        current_frame = cv.imread(frame_path + f"/result_{i + 1}.png")
        out.write(current_frame)

    out.release()
    cv.destroyAllWindows()

main()