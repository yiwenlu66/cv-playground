import sys
sys.path.extend(['../hed/', '../video'])

from analyze import frame_generator
from hed import generate
import cv2

model_path, video_file = sys.argv[1], sys.argv[2]

out = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))

heatmap_generator = generate(model_path)

for frame in frame_generator(video_file):
    heatmap_generator.next()
    heatmap = (heatmap_generator.send(frame) * 255).astype('uint8')
    ret, thresh = cv2.threshold(heatmap, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        BLUE = (255, 0, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), BLUE, 2)
    out.write(frame)


heatmap_generator.close()
out.release()