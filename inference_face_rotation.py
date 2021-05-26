import errno

import numpy as np
import cv2, os, audio
import subprocess
from tqdm import tqdm
import torch, torchvision
from models import Wav2Lip
import platform
import time
import pickle

import face_alignment

from skimage import transform as trans

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

path = os.path.abspath(__file__).rsplit("/", 1)[0]
face_transformation_reference = np.load(os.path.join(path, "FFHQ_template.npy")) / 4


def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i: i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes


def landmark_detection(images, pads=[0, 20, 5, 5]):
    pady1, pady2, padx1, padx2 = pads

    print("Landmark detection...")
    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)

    predictions_bb = []
    predictions_landmarks = []
    # crop_o_faces = []

    for i in tqdm(range(0, len(images)), ascii=True):
        img = images[i]
        pred_bb = detector.face_detector.detect_from_image(img)
        if pred_bb is None and i > 0:
            print("No Face founded. Using previous frame")
            img = images[-1]
            pred_bb = prev_bb
        elif if pred_bb is None and i == 0:
            cv2.imwrite('temp/faulty_frame.jpg', images[i])
            raise ValueError("No face on first frame of video. Image: 'temp/faulty_frame.jpg'")

        pred_landmark = detector.get_landmarks_from_image(img, pred_bb)[0]
        temp = np.array([pred_landmark[45], pred_landmark[42], pred_landmark[36], pred_landmark[39], pred_landmark[33]])

        predictions_landmarks.append(temp)
        rect = pred_bb[0].astype(int)
        pady2 = rect[3] // 11 - rect[1] // 11
        y1 = int(max(0, rect[1] - pady1))
        y2 = int(min(img.shape[0], rect[3] + pady2))
        x1 = int(max(0, rect[0] - padx1))
        x2 = int(min(img.shape[1], rect[2] + padx2))

        predictions_bb.append([y1, y2, x1, x2])
        prev_bb = pred_bb.copy()
        # crop_o_faces.append(img[y1:y2, x1:x2])

    del detector
    return np.array(predictions_landmarks), np.array(predictions_bb)  # , np.array(crop_o_faces)


def face_align(images, landmarks):
    out_size = (256, 256)
    print("Face transformation...")
    crop_images = []
    inv_params = []
    for i in tqdm(range(0, len(images)), ascii=True):
        img, source = images[i], landmarks[i]
        tform = trans.SimilarityTransform()
        tform.estimate(source, face_transformation_reference)
        M = tform.params[0:2, :]

        crop_img = cv2.warpAffine(img, M, out_size)
        crop_images.append(crop_img)

        tform2 = trans.SimilarityTransform()
        tform2.estimate(face_transformation_reference, source)
        inv_params.append(tform2.params[0:2, :])

    return crop_images, inv_params


def face_reverse_align(full_images, images, inverse_params):
    # print("Face inverse transformation...")
    new_images = []
    for f_img, img, inv_M in zip(full_images, images, inverse_params):
        h, w, _ = f_img.shape
        inv_crop_img = cv2.warpAffine(img, inv_M, (w, h))

        mask = np.ones((256, 256, 3), dtype=np.float32)  # * 255

        inv_mask = cv2.warpAffine(mask, inv_M, (w, h))
        inv_mask_erosion_removeborder = cv2.erode(inv_mask, np.ones((2, 2), np.uint8))  # to remove the black border

        inv_crop_img_removeborder = inv_mask_erosion_removeborder * inv_crop_img
        total_face_area = np.sum(inv_mask_erosion_removeborder) // 3
        w_edge = int(total_face_area ** 0.5) // 20  # compute the fusion edge based on the area of face
        erosion_radius = w_edge * 2
        inv_mask_center = cv2.erode(inv_mask_erosion_removeborder, np.ones((erosion_radius, erosion_radius), np.uint8))
        blur_size = w_edge * 2
        inv_soft_mask = cv2.GaussianBlur(inv_mask_center, (blur_size + 1, blur_size + 1), 0)
        merge_img = inv_soft_mask * inv_crop_img_removeborder + (1 - inv_soft_mask) * f_img
        new_images.append(merge_img.astype(np.uint8))

    return new_images


def face_detect(images, nosmooth=True, pads=[0, 20, 0, 0]):
    print("Face detection...")

    detector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device=device)
    predictions = []
    results = []
    pady1, pady2, padx1, padx2 = pads
    for i in tqdm(range(0, len(images)), ascii=True):
        predictions.extend(detector.face_detector.detect_from_image(images[i]))
        rect = predictions[-1]
        image = images[i]
        if rect is None and i == 0:
            cv2.imwrite('temp/faulty_frame.jpg', image)  # check this frame where the face was not detected.
            raise ValueError('Face not detected on first frame! Ensure the video contains a face in all the frames.')
        elif rect is None and i > 0:
            image = images[i - 1]
            rect = prev_rect

        pady2 = rect[3] // 11 - rect[1] // 11
        y1 = int(max(0, rect[1] - pady1))
        y2 = int(min(image.shape[0], rect[3] + pady2))
        x1 = int(max(0, rect[0] - padx1))
        x2 = int(min(image.shape[1], rect[2] + padx2))
        prev_rect = rect.copy()
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results


def datagen(frames, mels, inv_params, original_frames, original_bb, wav2lip_batch_size=128, img_size=96,
            pads=[0, 20, 0, 0]):
    img_batch, mel_batch, frame_batch, coords_batch, params_batch = [], [], [], [], []
    original_frames_batch, original_bb_batch = [], []

    face_det_results = face_detect(frames, pads=pads)  # BGR2RGB for CNN face detection

    for i, m in enumerate(mels):
        idx = i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (img_size, img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)
        params_batch.append(inv_params[idx])
        original_frames_batch.append(original_frames[idx])
        original_bb_batch.append(original_bb[idx])

        if len(img_batch) >= wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch, params_batch, original_frames_batch, original_bb_batch
            img_batch, mel_batch, frame_batch, coords_batch, params_batch = [], [], [], [], []
            original_frames_batch, original_bb_batch = [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch, params_batch, original_frames_batch, original_bb_batch


def _load(checkpoint_path):
    if 'cuda' in device:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()


def generate_video(face_path, audio_path, outfile, checkpoint_path='/root/Wav2Lip/checkpoints/wav2lip_gan.pth',
                   resize_factor=1, crop=[0, -1, 0, -1], pads=[0, 20, 0, 0], wav2lip_batch_size=128,
                   draw_bb=False, store_bb=False, raw_output=False):
    if not os.path.isfile(face_path):
        raise ValueError('--face argument must be a valid path to video file')
    else:
        s = time.time()
        video_stream = cv2.VideoCapture(face_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            y1, y2, x1, x2 = crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)
        print('Reading video time:', time.time() - s)

    print("Number of frames available for inference: " + str(len(full_frames)))

    if not audio_path.endswith('.wav'):
        print('Extracting raw audio...')
        command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_path, 'temp/temp.wav')

        subprocess.call(command, shell=True)
        audio_path = os.path.join(path, 'temp/temp.wav')

    # loading audio
    wav = audio.load_wav(audio_path, 16000)
    mel = audio.melspectrogram(wav)

    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    print("Length of mel chunks: {}".format(len(mel_chunks)))

    full_frames = full_frames[:len(mel_chunks)]  # remove excess frames

    batch_size = wav2lip_batch_size
    if os.path.isfile(face_path.rsplit('.', 1)[0] + '_landmarks.pickle'):
        print("Using preprocessed landmarks...")
        with open(face_path.rsplit('.', 1)[0] + '_landmarks.pickle', 'rb') as f:
            landmarks = pickle.load(f)
            landmarks, original_faces_bb = landmarks[0][:len(mel_chunks)], landmarks[1][:len(mel_chunks)]
    else:
        landmarks, original_faces_bb = landmark_detection(full_frames.copy())

    crop_faces = []
    for frame, bb in zip(full_frames.copy(), original_faces_bb):
        crop_faces.append(frame[bb[0]:bb[1], bb[2]:bb[3]])

    crop_frames, inverse_params = face_align(full_frames.copy(), landmarks)

    if store_bb:
        with open(outfile.rsplit('.', 1)[0] + '.pickle', 'wb') as f:
            pickle.dump([original_faces_bb, crop_faces], f)

    gen = datagen(crop_frames.copy(), mel_chunks, inverse_params, full_frames, original_faces_bb, pads=pads)

    s = time.time()
    for i, (img_batch, mel_batch, frames, coords, params, o_frames, o_bb) in enumerate(
            tqdm(gen, total=int(np.ceil(float(len(mel_chunks)) / batch_size)))):
        if i == 0:
            model = load_model(checkpoint_path)
            print("Model loaded")

            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi',
                                  cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
            if raw_output:
                out_raw = cv2.VideoWriter('temp/result_raw.avi',
                                          cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        # prediction
        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c, ip, of, oc in zip(pred, frames, coords, params, o_frames, o_bb):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

            # replace only lips
            # y_ = y1 + int((y2 - y1) / 2)
            # f[y_:y2, x1:x2] = p[p.shape[0] // 2:, :]
            # replace full face
            f[y1:y2, x1:x2] = p

            f = face_reverse_align([of], [f], [ip])[0]

            # draw bounding box of face and lips
            if draw_bb:
                y1, y2, x1, x2 = oc
                y_ = y1 + int((y2 - y1) / 2)
                cv2.line(f, (x1, y1), (x1, y_), (0, 0, 255), 2)
                cv2.line(f, (x2, y1), (x2, y_), (0, 0, 255), 2)
                cv2.line(f, (x1, y1), (x2, y1), (0, 0, 255), 2)

                cv2.line(f, (x1, y_), (x1, y2), (0, 255, 0), 2)
                cv2.line(f, (x2, y_), (x2, y2), (0, 255, 0), 2)
                cv2.line(f, (x1, y2), (x2, y2), (0, 255, 0), 2)
                cv2.line(f, (x1, y1 + int((y2 - y1) / 2)), (x2, y1 + int((y2 - y1) / 2)), (0, 255, 0), 2)
            out.write(f)

            if raw_output:
                f_raw = np.ones(of.shape, dtype=np.uint8)
                y1, y2, x1, x2 = oc
                f_raw[:, :, 1] = 255
                f_raw[y1:y2, x1:x2] = f[y1:y2, x1:x2]
                out_raw.write(f_raw)

    print("Final time:", time.time() - s)
    out.release()

    command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result.avi', outfile)
    subprocess.call(command, shell=platform.system() != 'Windows')

    if raw_output:
        out_raw.release()
        command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_path, 'temp/result_raw.avi',
                                                                      outfile.rsplit('.', 1)[0] + "_raw.mp4")
        subprocess.call(command, shell=platform.system() != 'Windows')

    return outfile


def video_face_detection(face_path, resize_factor=1, crop=[0, -1, 0, -1], nosmooth=True, pads=[0, 20, 0, 0]):
    if not os.path.isfile(face_path):
        raise ValueError('--face argument must be a valid path to video/image file')
    else:
        s = time.time()
        video_stream = cv2.VideoCapture(face_path)

        print('Reading video frames...')

        full_frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor))

            y1, y2, x1, x2 = crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]

            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)
        print('Reading video time:', time.time() - s)
    print("Number of frames available for inference: " + str(len(full_frames)))

    face_landmarks, face_bb = landmark_detection(full_frames.copy())
    # crop_images, inv_params = face_align(full_frames.copy(), face_landmarks)

    # face_det_results = face_detect(crop_images, nosmooth, pads=pads)

    with open(face_path.rsplit('.', 1)[0] + '_landmarks.pickle', 'wb') as f:
        pickle.dump([face_landmarks, face_bb], f)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--face_detection', action='store_true', default=False,
                        help='run face detection for a video')
    parser.add_argument('--draw_bb', action='store_true', default=False,
                        help='draw red bounding box for lips and face')

    parser.add_argument('--store_bb', action='store_true', default=False,
                        help='store bounding boxes')

    parser.add_argument('--wav2lip_gan', action='store_true', default=False,
                        help='store bounding boxes')

    parser.add_argument('--raw_output', action='store_true', default=False,
                        help='raw output')

    parser.add_argument('--prep', action='store_true', default=False,
                        help='preprocess video')

    parser.add_argument('--audio', default=False, required=False, type=str,
                        help='path to audio')
    parser.add_argument('--video', default=False, required=True, type=str,
                        help='path to video')
    parser.add_argument('--output', default=None, required=False, type=str,
                        help='path to video')

    args = parser.parse_args()

    if args.face_detection:
        video_face_detection(args.video)

    else:
        if args.wav2lip_gan:
            checkpoint_path = os.path.join(path, "checkpoints/wav2lip_gan.pth")
        else:
            checkpoint_path = os.path.join(path, "checkpoints/wav2lip.pth")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_path)

        if not os.path.isfile(args.video.rsplit('.', 1)[0] + '_landmarks.pickle') or args.prep:
            video_face_detection(args.video)

        if args.output:
            generate_video(args.video, args.audio, args.output, draw_bb=args.draw_bb,
                           store_bb=args.store_bb, raw_output=args.raw_output, checkpoint_path=checkpoint_path)
        else:
            generate_video(args.video, args.audio, "temp/" + args.video.split('/')[-1], draw_bb=args.draw_bb,
                           store_bb=args.store_bb, raw_output=args.raw_output, checkpoint_path=checkpoint_path)
