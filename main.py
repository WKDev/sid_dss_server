# https://awakening95.tistory.com/1 socket
# https://surprisecomputer.tistory.com/22 pi to pc socket
import threading
from anyio import run_sync_from_thread
import uvicorn
from stream_handler import to_b64
import time
import requests
import base64
import zmq
from pydantic import BaseModel
import cv2
from typing import Union
from starlette.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI, WebSocket
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from models.common import DetectMultiBackend
import os
import sys
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import asyncio
import numpy as np

from PIL import ImageFont, ImageDraw, Image

EXT_URL = 'http://192.168.1.100:5000/cmd/ext'

# BIRD_CAM =
GROWTH_TRACKING_CAM = 'http://192.168.1.101:5000/1'
BUG_DETECTION_CAM = 'http://192.168.1.101:5000/2'

BIRD_DET_1 = "조류퇴치_1"
BIRD_DET_2 = "조류퇴치_2"
BIRD_DET_ERR = "조류퇴치_Check Connection again."

BUG_TITLE = "노린재 트랩"

# from blobdetection import blobdetector

global ROOT

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


MAX_FPS = 100

app = FastAPI()
ws_app = FastAPI()


origins = [
    "http://127.0.0.1"
    "http://127.0.0.1:3000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
ws_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


global inferenced_img1
global inferenced_img2
global inferenced_img3
inferenced_img1 = None
inferenced_img2 = None
inferenced_img1 = None
inferenced_img2 = None

global enable_extermination
global last_executed
global enable_bird_overlay
global enable_bugdetection
global enable_bug_overlay

global show_demo_bird
global show_demo_bug
global stop_demo_bird
global curr_mode  # demo or inf mode
curr_mode = 0


global is_bird
global is_bug

is_bird = False
is_bug = False


enable_extermination = 1
enable_bugdetection = 1

enable_bird_overlay = 1
enable_bug_overlay = 1

show_demo_bird = 0
show_demo_bug = 0


def blobdetector(source):
    global enable_bugdetection
    global show_demo_bug
    global is_bug

    detected_time = time.time()
    blink_time = time.time()
    warning_timer = False
    red_img = np.full((480, 640, 3), (0, 0, 255), dtype=np.uint8)

    # Load image
    # image = cv2.imread('blob.png', 0)

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea = 15
    params.maxArea = 200
    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    
    params.minConvexity = 0.2

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    demo_source = 0

    while True:
        if show_demo_bug:
            print('demo stream')
            cap = cv2.VideoCapture('/home/ioss/ioss/backend/assets/demo/bug/bug_demo.mp4')
            show_demo_bug = 0

        else:
            # print('real stream')

            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                cap.release()
                target_img = cv2.imread('assets/bug_opening.png')

                ret, preprocessed = cv2.imencode('.jpg', target_img)

                raw_img = preprocessed.tobytes()

                yield (b'--frame\r\n'
                        b'Content-Type:image/jpeg\r\n'
                        b'Content-Length: ' +
                        f"{len(raw_img)}".encode() + b'\r\n'
                        b'\r\n' + bytearray(raw_img) + b'\r\n')


            show_demo_bug = 0



        while cap.isOpened():
            ret, frame = cap.read()

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            red_img = np.full((h, w, 3), (0, 0, 255), dtype=np.uint8)



            if demo_source == 0 and show_demo_bug == 1:
                cap.release()
                demo_source = 1
                print('stop realstream')
                break

            if demo_source == 1 and show_demo_bug == 1:
                cap.release()
                demo_source = 0
                show_demo_bug = 0
                print('stop bugstream')
                break

            if demo_source ==1:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


            if ret:

                if enable_bugdetection:
                    # print('bugdetection enabled')
                    cvted_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

                    # Detect blobs
                    keypoints = detector.detect(cvted_frame)

                    # Draw blobs on our image as red circles
                    blank = np.zeros((1, 1))
                    ret_detected = cv2.drawKeypoints(cvted_frame, keypoints, blank, (0, 0, 255),
                                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                    number_of_blobs = len(keypoints)
                    text = "Bugs Detected : " + str(len(keypoints))
                    
                    ret_detected = putKorean(ret_detected, BUG_TITLE, (20,40))


                    if number_of_blobs:
                        cv2.putText(ret_detected, text, (80, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                    if number_of_blobs > 2 and abs(time.time()-detected_time) > 3:
                        is_bug = True
                        detected_time = time.time()
                        # print('BUG DETECTED')

                    if abs(time.time()-detected_time) < 2:
                        # try:
                        ret_detected = cv2.addWeighted(
                            ret_detected, 0.8, red_img, 0.3, 0)

                        # except:
                        # print('userWarning : image shapes are not matched!!')

                    # cv2.imshow("Filtering Circular Blobs Only", blobs)

                    ret, img = cv2.imencode('.jpg', ret_detected)

                    byte_img = img.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type:image/jpeg\r\n'
                           b'Content-Length: ' +
                           f"{len(byte_img)}".encode() + b'\r\n'
                           b'\r\n' + bytearray(byte_img) + b'\r\n')

                else:
                    target_img = cv2.imread('assets/bug_opening.png')

                    ret, preprocessed = cv2.imencode('.jpg', target_img)

                    raw_img = preprocessed.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type:image/jpeg\r\n'
                           b'Content-Length: ' +
                           f"{len(raw_img)}".encode() + b'\r\n'
                           b'\r\n' + bytearray(raw_img) + b'\r\n')
            else:

                    target_img = cv2.imread('assets/bug_opening.png')

                    ret, preprocessed = cv2.imencode('.jpg', target_img)

                    raw_img = preprocessed.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type:image/jpeg\r\n'
                           b'Content-Length: ' +
                           f"{len(raw_img)}".encode() + b'\r\n'
                           b'\r\n' + bytearray(raw_img) + b'\r\n')
            


def putKorean(img, text, pos):
    # img = np.full(shape=(480,640,3), fill_value=255, dtype=np.uint8)
    img = Image.fromarray(img)
    font = ImageFont.truetype("fonts/gulim.ttc", 40)
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, font=font, fill=(0,0,0))
    return np.array(img)

    

def image_to_byte(img_id=1):
    global inferenced_img1
    global inferenced_img2
    global inferenced_img3

    while True:

        if img_id == 1 and inferenced_img1 is not None:
            target_img = inferenced_img1
        elif img_id == 2 and inferenced_img2 is not None:
            target_img = inferenced_img2

        else:
            if img_id ==1:
                target_img = cv2.imread('assets/cam1_opening.png')
            elif img_id==2:
                target_img = cv2.imread('assets/cam2_opening.png')

            # cv2.putText(target_img, BIRD_DET_ERR, (20, 40),
            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            
            target_img = putKorean(target_img, BIRD_DET_ERR, (20,40))
            # print('cannot detect cam!' + 'img_id')
            # print(inferenced_img1)
            # print(inferenced_img2)
            ret, buffer = cv2.imencode('.jpg', target_img)

            # frame을 byte로 변경 후 특정 식??으로 변환 후에
            # yield로 하나씩 넘겨준다.
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(frame) + b'\r\n')

            break

        ret, buffer = cv2.imencode('.jpg', target_img)

        # frame을 byte로 변경 후 특정 식??으로 변환 후에
        # yield로 하나씩 넘겨준다.
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(frame) + b'\r\n')


frontend_root = ROOT / "../client/build"
static_root = ROOT / "../client/build/static"


class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        if response.status_code == 404:
            response = await super().get_response('.', scope)
        return response


app.mount("/static", StaticFiles(directory=static_root), name="static")
app.mount('/static/', SPAStaticFiles(directory=static_root,
          html=True), name='static')


@app.get("/")
def index():
    frontend_root = ROOT / "../client/build"

    return FileResponse(str(frontend_root) + '/index.html', media_type='text/html')


@app.get("/inf/1")  # inferenced router for bird_detection
async def inferenced_1():
    return StreamingResponse(image_to_byte(img_id=1), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/inf/2")  # inferenced router for bird_detection
def inferenced_2():
    return StreamingResponse(image_to_byte(img_id=2), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/bug")  # inferenced router for bird_detection
def inference_bug():

    global enable_bugdetection

    return StreamingResponse(blobdetector(source=BUG_DETECTION_CAM), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/inf/en_ext/{cmd_id}")
def set_inference(cmd_id: int, q: Union[str, None] = None):
    global enable_extermination
    enable_extermination = cmd_id
    return 202


@app.get("/inf/en_overlay/{cmd_id}")
def set_inference(cmd_id: int, q: Union[str, None] = None):
    global enable_bird_overlay
    enable_bird_overlay = cmd_id
    return 202


@app.get("/bug/en_bug/{cmd_id}")
def set_inference(cmd_id: int, q: Union[str, None] = None):
    global enable_bugdetection
    enable_bugdetection = cmd_id
    print("bugdetection enabled")
    return 202


@app.get("/bug/en_overlay/{cmd_id}")
def set_inference(cmd_id: int, q: Union[str, None] = None):
    global enable_bug_overlay
    enable_bug_overlay = cmd_id
    return 202


@app.get("/inf/ext_force")
def force_extermination():
    send_extermination(execution_time=-1)

    return 202


@app.get("/cmd/dummy-bird")
def bummy_bird():
    global show_demo_bird

    if show_demo_bird == 0:
        show_demo_bird = 1
    elif show_demo_bird == 1:
        show_demo_bird = 0

    return 202


@app.get("/cmd/dummy-bug")
def bummy_bug():
    global show_demo_bug

    if show_demo_bug == 0:
        show_demo_bug = 1
    elif show_demo_bug == 1:
        show_demo_bug = 0

    return 202


@app.websocket("/notification")
async def websocket_endpoint(websocket: WebSocket):
    global enable_extermination
    global enable_bugdetection
    global enable_bird_overlay
    global enable_bug_overlay
    global is_bird
    global is_bug

    print(f"notification channel connected : {websocket.client}")
    await websocket.accept()  # client의 websocket접속 허용
    await websocket.send_json({'code': '{}'.format(202), 'style': 'primary', 'toast_title': 'Alert', 'toast_msg': 'Successfully Connected.', 'time': '{}'.format(time.time())})

    init_json = {'code': str(101), 'en_ext': str(enable_extermination), 'ext_overlay': str(
        enable_bird_overlay), 'en_bug': str(enable_bugdetection), 'bug_overlay': str(enable_bug_overlay)}

    await websocket.send_json(init_json)

    while True:
        # data = await websocket.receive_text()  # client 메시지 수신대기
        # print('[{}{}] while loop is running in websocket...'.format(is_bird,is_bug))
        if is_bird:
            await websocket.send_json({'code': str(1), 'style': 'warning', 'toast_title': 'Alert', 'toast_msg': 'Birds are detected.', 'time': str(time.time())})
            # print('sending bird noti')
            is_bird = False

        if is_bug:
            await websocket.send_json({'code': str(1), 'style': 'danger', 'toast_title': 'Alert', 'toast_msg': 'Bugs are detected.', 'time': str(time.time())})
            # print('sending bug noti')

            is_bug = False

        await asyncio.sleep(0.25)


def send_extermination(num_objects=0, length=3, execution_time=0):
    global is_bird
    global last_executed
    time_interval = (execution_time - last_executed)

    json_data = {'name': 'sidserver', 'length': length}

    if num_objects > 5 and time_interval > 10:
        is_bird = True
        print('EXTERMINATING...' + str(is_bird))
        # print('time elasped since last run : {} num_objects : {} length : {} event_time {}'.format(
        #     time_interval, num_objects, length, execution_time))

        # Send the data.
        try:
            response = requests.post(url=EXT_URL, json=json_data)
            print("rpi responded with %s" % response.status_code)

        except:
            pass

        last_executed = execution_time

    if execution_time == -1:
        print('force command')
        # Send the data.
        try:
            response = requests.post(url=EXT_URL, json=json_data)
            print("rpi responded with %s" % response.status_code)

        except:
            pass

        last_executed = execution_time


@torch.no_grad()
def run(
        weights=ROOT / 'models/sid_bird_220523.pt',  # model.pt path(s)
        source='cameras.txt',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/sid_bird_220523.yaml',  # dataset.yaml path
        imgsz=(640, 480),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=True,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=True,  # hide labels
        hide_conf=True,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):

    global inferenced_img1
    global inferenced_img2
    global enable_extermination
    global enable_bird_overlay

    global is_bird

    global show_demo_bird
    global show_demo_bug
    global stop_demo_bird
    global curr_mode
    demo_file = Path(__file__).resolve()
    demo_root = demo_file.parents[0]  # YOLOv5 root directory
    if str(demo_root) not in sys.path:
        sys.path.append(str(demo_root))  # add ROOT to PATH
    demo_root = Path(os.path.relpath(demo_root, Path.cwd()))  # relative

    while True:

        if show_demo_bird:
            demo_path = demo_root / '/assets/demo'
            source = str('/home/ioss/ioss/backend/assets/demo/bird')
            print('/home/ioss/ioss/backend/assets/demo')
            print('demo bird enbled.')
            show_demo_bird = 0
        else:
            source = str('cameras.txt')
            show_demo_bird = 0
            print('typical run')

        save_img = not nosave and not source.endswith(
            '.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith(
            '.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(
            Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True,
                                                              exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(
            weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz,
                                  stride=stride, auto=pt)
            bs = len(dataset)  # batch_size
        else:
            dataset = LoadImages(source, img_size=imgsz,
                                 stride=stride, auto=pt)
            bs = 1  # batch_size
        vid_path, vid_writer = [None] * bs, [None] * bs

        model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0], 0

        for path, im, im0s, vid_cap, s in dataset:
            time.sleep(0.05)

            if curr_mode == 0 and show_demo_bird:
                print('demo stream started.')
                curr_mode = 1
                dataset.release()
                break

            if curr_mode == 1 and show_demo_bird:
                print('demo stream stopped.')
                show_demo_bird = 0
                curr_mode = 0
                break

            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            visualize = increment_path(
                save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            dt[2] += time_sync() - t3

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, frame = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + \
                    ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
                s += '%gx%g ' % im.shape[2:]  # print string
                # normalization gain whwh
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
                imc = im0.copy() if save_crop else im0  # for save_crop

                if enable_extermination == 1:
                    annotator = Annotator(
                        im0, line_width=line_thickness, example=str(names))
                    if len(det) > 3:
                        if enable_extermination == 1:
                            send_extermination(num_objects=len(
                                det), length=3, execution_time=time.time())

                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            # add to string
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(
                                    1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                # label format
                                line = (
                                    cls, *xywh, conf) if save_conf else (cls, *xywh)
                                with open(f'{txt_path}.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() %
                                            line + '\n')

                            if save_img or save_crop or view_img:  # Add bbox to image
                                c = int(cls)  # integer class
                                label = None if hide_labels else (
                                    names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                annotator.box_label(
                                    xyxy, label, color=colors(c, True))
                            if save_crop:
                                save_one_box(
                                    xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

                    # Stream results
                    im0 = annotator.result()

                # if view_img:
                #     cv2.imshow(str(p), im0)
                #     cv2.waitKey(1)  # 1 millisecond


                cv2.putText(im0, f'{1/(t3 - t2):.3f}fps', (510, 460),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # target_img, text_content, position(x,y), font, ?,color, ?
                if str(p)[-1] == '1':
                    inferenced_img1 = im0
                    
                    # cv2.putText(inferenced_img1, BIRD_DET_1, (20, 40),
                    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    inferenced_img1 = putKorean(inferenced_img1, BIRD_DET_1, (20,40))


                # encoded, buffer = cv2.imencode('.jpg', inferenced_img1)
                # footage_socket.send_string(base64.b64encode(buffer))
                elif str(p)[-1] == '2':

                    # cv2.putText(inferenced_img2, BIRD_DET_2, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    inferenced_img2 = im0
                    inferenced_img2 = putKorean(inferenced_img2, BIRD_DET_2, (20,40))

                else:
                    # only occurs in case of demo
                    inferenced_img1 = im0
                    inferenced_img2 = im0

            # Print time (inference-only)
            # LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        print('escape from for loop')


if __name__ == "__main__":
    # CHECK INITIAL TIME FOR PREVENT NUMEROUS EXTERMINATION REQUEST
    last_executed = time.time()

    check_requirements(exclude=('tensorboard', 'thop'))
    t = threading.Thread(target=run)
    t.start()

    uvicorn.run(app, host="0.0.0.0", port=8080)
