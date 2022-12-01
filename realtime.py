import math

import cv2
import tensorflow as tf
from tensorflow import keras
from metrics import iou
from datetime import datetime
import numpy as np
import time
from threading import Thread


class ThreadedCamera(object):
    def __init__(self, source = 0):

        self.capture = cv2.VideoCapture(source)

        self.thread = Thread(target = self.update, args = ())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame  = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def grab_frame(self):
        if self.status:

            return self.frame
        return None

def load_tf_model(path):
    with tf.keras.utils.CustomObjectScope({'iou': iou}):
        model = tf.keras.models.load_model(path)
        # model = tf.saved_model.load(path)
        return model

def create_label_colormap():
    """Creates a label colormap used in Cityscapes segmentation benchmark.

    Returns:
        A Colormap for visualizing segmentation results.
    """

    #B,G,R
    colormap = np.array([
        [0,  0, 0],
        [153,  0, 0],
        [0, 0,  255],
        [0, 255, 255]], dtype=np.uint8)
    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
        label: A 2D array with integer type, storing the segmentation label.

    Returns:
        result: A 2D array with floating type. The element of the array
            is the color indexed by the corresponding element in the input label
            to the PASCAL color map.

    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
            map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

# def jarak_antar_pixel(koordinat_ban, koordinat_marka):
#     jarak = np.abs((koordinat_ban - koordinat_marka)*0.0353)
#
#     return jarak

def jarak_antar_pixel(koordinat_ban, koordinat_marka):

    if (koordinat_marka == 401) | (koordinat_marka == -1):
        jarak = "???"

        return jarak

    else:
        jarak = np.abs((koordinat_ban - koordinat_marka)*1.7857142857) #konstanta 1.7857142857

    return jarak



if __name__ == '__main__':
    stream_link = "FIx Testing/Ban kanan garis sambung 0 m (1).mp4" #stream dashcam ("tcp://193.168.0.1:6200/"))
    streamer = ThreadedCamera(stream_link)

    model_path = "model10kepoch.h5"
    model = load_tf_model(model_path)

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        frame = streamer.grab_frame()
        if frame is not None:
            H, W = frame.shape[:2]
            # image = cv2.imread(frame, 0)
            # H, W, _ = image.shape
            image = cv2.resize(frame, (128,128))  ## (256, 256, 3)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = np.array(image)
            test_image = np.expand_dims(image, axis=2)
            test_image = tf.keras.utils.normalize(test_image, axis=1)
            test_img_norm = test_image[:, :, 0][:, :, None]
            test_img_input = np.expand_dims(test_img_norm, 0)
            # image = image / 255.0
            # image = np.expand_dims(image, axis=0)       ## (1, 256, 256, 3)
            # image = image.astype(np.float32)

            """ Predict """
            start_time = time.time()
            prediction = (model.predict(test_img_input))
            predicted_img = np.argmax(prediction, axis=3)[0, :, :]

            # mask = cv2.resize(predicted_img, (128, 128))
            #

            # print(predicted_img.shape)
            mask_ori = cv2.resize(predicted_img.astype(np.uint8), (H, W))

            # print((mask_ori.shape))

            seg_image = label_to_color_image(predicted_img).astype(np.uint8)
            # mask = mask > 0.5
            # mask = mask * 255
            # mask = mask.astype(np.float32)

            ############IPM-MASK#############
            # pts1_mask = np.float32([[0, 1400], [2560, 1400], [1000, 950], [1560, 950]])
            # pts2_mask = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
            # matrix_mask = cv2.getPerspectiveTransform(pts1_mask, pts2_mask)
            #
            # result_warp_mask = cv2.warpPerspective(mask_ori, matrix_mask, (400, 600))
            # result_flipped_mask = cv2.flip(result_warp_mask, 0)
            ############################

            now = datetime.now()

            # convert to string
            date_time_str = now.strftime("%Y%m%d_%H%M%S")
            # print('DateTime String:', date_time_str)

            save_path_mask = f"resultsMask/{date_time_str}.png"
            # cv2.imwrite(save_path_mask, seg_image)

            # predicted_img_overlay = mask.astype(np.uint8)
            # predicted_img_overlay = np.dstack([predicted_img_overlay, predicted_img_overlay, predicted_img_overlay])
            ori_seg_img = cv2.resize(seg_image, (W, H))

            ############ IPM-MASK 2560X1600 #############
            # pts1_mask_ori = np.float32([[0, 1300], [2559, 1400], [1000, 1050], [1700, 1050]])
            # # pts1_mask_ori = np.float32([[0, 1300], [2559, 1400], [1000, 1050], [1700, 1050]])  ###terdistorsi
            # pts2_mask_ori = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
            # matrix_mask_ori = cv2.getPerspectiveTransform(pts1_mask_ori, pts2_mask_ori)
            #
            # result_warp_ori = cv2.warpPerspective(ori_seg_img, matrix_mask_ori, (400, 600))
            # result_flipped_ori = cv2.flip(result_warp_ori, 0)

            ##############################Distorsi Kanan Ban kiri 0,5 m + 100  ###############
            pts1_mask_ori = np.float32([[200, 1300], [2759, 1400], [1200, 1050], [1900, 1050]])  ###terdistorsi
            pts2_mask_ori = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
            matrix_mask_ori = cv2.getPerspectiveTransform(pts1_mask_ori, pts2_mask_ori)

            result_warp_ori = cv2.warpPerspective(ori_seg_img, matrix_mask_ori, (400, 600))
            result_flipped_ori = cv2.flip(result_warp_ori, 0)
            ############################

            ############ IPM-MASK 800X480 #############
            # pts1_mask_ori = np.float32([[0, 420], [800, 420], [300, 285], [468, 285]])
            # pts2_mask_ori = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
            # matrix_mask_ori = cv2.getPerspectiveTransform(pts1_mask_ori, pts2_mask_ori)
            #
            # result_warp_ori = cv2.warpPerspective(ori_seg_img, matrix_mask_ori, (400, 600))
            # result_flipped_ori = cv2.flip(result_warp_ori, 0)
            ############################

            # indices = np.where(np.all(a == c, axis=-1))
            # print(indices)
            # print(zip(indices[0], indices[1]))


            # print(ori_img.shape)
            # print(frame.shape)
            overlay = cv2.addWeighted(frame,1.0, ori_seg_img,0.5,0)

            ##########INVERSE-PERSPECTIVE-MAPPING-OVERLAY 2560X1600##################

            # cv2.line(frame, (0, 1400), (2560, 1400), (0, 0, 255), 3)
            # cv2.line(frame, (0, 1400), (1000, 950), (0, 0, 255), 3)
            # cv2.line(frame, (1560, 950), (2560, 1400), (0, 0, 255), 3)
            # cv2.line(frame, (1560, 950), (1000, 950), (0, 0, 255), 3)

            # pts1_overlay = np.float32([[0, 1400], [2560, 1400], [1000, 950], [1560, 950]])
            # # pts1_overlay = np.float32([[50, 1450], [2610, 1450], [1050, 1000], [1610, 1000]]) ###terdistorsi
            # pts2_overlay = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
            # matrix_overlay = cv2.getPerspectiveTransform(pts1_overlay, pts2_overlay)
            #
            # result_warp_overlay = cv2.warpPerspective(overlay, matrix_overlay, (400, 600))
            # result_flipped_overlay = cv2.flip(result_warp_overlay, 0)

            # ########################## EDITED ########################
            # pts1_overlay = np.float32([[0, 1300], [2559, 1400], [1000, 1050], [1700, 1050]])
            # # pts1_overlay = np.float32([[50, 1450], [2610, 1450], [1050, 1000], [1610, 1000]]) ###terdistorsi
            # pts2_overlay = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
            # matrix_overlay = cv2.getPerspectiveTransform(pts1_overlay, pts2_overlay)
            #
            # result_warp_overlay = cv2.warpPerspective(overlay, matrix_overlay, (400, 600))
            # result_flipped_overlay = cv2.flip(result_warp_overlay, 0)

            ########################## EDITED + 50  ########################
            pts1_overlay = np.float32([[200, 1300], [2759, 1400], [1200, 1050], [1900, 1050]])
            # pts1_overlay = np.float32([[50, 1450], [2610, 1450], [1050, 1000], [1610, 1000]]) ###terdistorsi
            pts2_overlay = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
            matrix_overlay = cv2.getPerspectiveTransform(pts1_overlay, pts2_overlay)

            result_warp_overlay = cv2.warpPerspective(overlay, matrix_overlay, (400, 600))
            result_flipped_overlay = cv2.flip(result_warp_overlay, 0)

            ########### INVERSE-PERSPECTIVE-MAPPING-OVERLAY 800X480#################
            # pts1_overlay = np.float32([[0, 420], [800, 420], [300, 285], [468, 285]])
            # pts2_overlay = np.float32([[0, 0], [400, 0], [0, 600], [400, 600]])
            # matrix_overlay = cv2.getPerspectiveTransform(pts1_overlay, pts2_overlay)
            #
            # result_warp_overlay = cv2.warpPerspective(overlay, matrix_overlay, (400, 600))
            # result_flipped_overlay = cv2.flip(result_warp_overlay, 0)


            # total_time = time.time() - start_time
            # time_taken.append(total_time)

            # save_path_vis = f"resultsVis/{date_time_str}.png"
            # cv2.imwrite(save_path_vis, overlay)

            ########### Lane Warning ##########

            # yellow = lm_dashed
            # red = lm_solid

            indicesyellow = np.where(np.all(result_flipped_ori == (0, 255, 255), axis=-1))
            indicesred = np.where(np.all(result_flipped_ori == (0, 0,  255), axis=-1))

            # print(indices)
            ################# Perhitungan Marka Garis Putus-Putus #########################
            # variable yang menampung koord y dan x
            list_x = indicesyellow[0]
            list_y = indicesyellow[1]

            list_x_kiri_dashed = []
            list_x_kanan_dashed = []

            list_y_kiri_dashed = []
            list_y_kanan_dashed = []

            for i in range(len(list_y)):
                if list_y[i] < 200:
                    list_y_kiri_dashed.append(list_y[i])
                    list_x_kiri_dashed.append(list_x[i])
                else:
                    list_y_kanan_dashed.append(list_y[i])
                    list_x_kanan_dashed.append(list_x[i])

            # print(len(list_x_kanan), len(list_y_kanan))
            # print(len(list_x_kiri), len(list_y_kiri))

            if len(list_x_kanan_dashed)!=0 & (len(list_y_kanan_dashed) != 0):
                xm_kanan_dash = np.amax(list_x_kanan_dashed)
                ym_kanan_dash = np.amax(list_y_kanan_dashed)

            else:
                xm_kanan_dash = 601
                ym_kanan_dash = 401

            if len(list_x_kiri_dashed) != 0 & (len(list_y_kiri_dashed) != 0):

                xm_kiri_dash = np.amax(list_x_kiri_dashed)
                ym_kiri_dash = np.amax(list_y_kiri_dashed)

            else:
                xm_kiri_dash = 601
                ym_kiri_dash = -1

            ban_kanan = 275  # koordinat y ban kanan
            ban_kiri = 190  #koordinat y ban kanan 185

            lajur_kiri = ym_kiri_dash
            if lajur_kiri > ban_kiri:
                cv2.circle(overlay, (100, 100), 100, (0, 255, 255), -1)
                print("Ban Kiri Hati-Hati")

            else:
                print("Ban Kiri Aman")

            lajur_kanan = ym_kanan_dash
            if lajur_kanan > ban_kanan:
                print("Ban Kanan Aman")

            else:
                print("Ban Kanan Hati-Hati")
                cv2.circle(overlay, (2400, 100), 100, (0, 255, 255), -1)

            ################# Perhitungan Marka Garis Sambung #########################
            # variable yang menampung koord y dan x
            list_x_red = indicesred[0]
            list_y_red = indicesred[1]

            list_x_kiri_solid = []
            list_x_kanan_solid = []

            list_y_kiri_solid = []
            list_y_kanan_solid = []

            for i in range(len(list_y_red)):
                if list_y_red[i] < 200:
                    list_y_kiri_solid.append(list_y_red[i])
                    list_x_kiri_solid.append(list_x_red[i])
                else:
                    list_y_kanan_solid.append(list_y_red[i])
                    list_x_kanan_solid.append(list_x_red[i])

            # print(len(list_x_kanan), len(list_y_kanan))
            # print(len(list_x_kiri), len(list_y_kiri))

            if len(list_x_kanan_solid) != 0 & (len(list_y_kanan_solid) != 0):
                xm_kanan_solid = np.amax(list_x_kanan_solid)
                ym_kanan_solid = np.amax(list_y_kanan_solid)

            else:
                xm_kanan_solid = 601
                ym_kanan_solid = 401

            if len(list_x_kiri_solid) != 0 & (len(list_y_kiri_solid) != 0):

                xm_kiri_solid = np.amax(list_x_kiri_solid)
                ym_kiri_solid = np.amax(list_y_kiri_solid)

            else:
                xm_kiri_solid = 601
                ym_kiri_solid = -1

            lajur_kiri_solid = ym_kiri_solid
            if lajur_kiri_solid > ban_kiri:
                cv2.circle(overlay, (100, 100), 100, (0, 0, 255), -1)
                print("Ban Kiri Hati-Hati")

            else:
                print("Ban Kiri Aman")

            lajur_kanan_solid = ym_kanan_solid
            if lajur_kanan_solid > ban_kanan:
                print("Ban Kanan Aman")

            else:
                print("Ban Kanan Hati-Hati")
                cv2.circle(overlay, (2400, 100), 100, (0, 0, 255), -1)


            ####### Kalkulasi Jarak ########
            jarak_ban_kiri_dashed = jarak_antar_pixel(ban_kiri, ym_kiri_dash)
            jarak_ban_kanan_dashed = jarak_antar_pixel(ban_kanan, ym_kanan_dash)

            jarak_ban_kiri_solid = jarak_antar_pixel(ban_kiri, ym_kiri_solid)
            jarak_ban_kanan_solid = jarak_antar_pixel(ban_kanan, ym_kanan_solid)
            # print(jarak_ban_kanan_dashded)


            ###### Display Configuration #########
            font = cv2.FONT_HERSHEY_SIMPLEX
            org_kiri_dashed = (100, 400)
            org_kanan_dashed = (100, 430)
            org_kiri_solid = (100, 460)
            org_kanan_solid = (100, 490)
            thickness = 1
            fontScale = 1
            color = (255, 255, 255)
            overlay_jarak_kiri = cv2.putText(overlay, "Jarak Ban Kiri Putus2: " + str(jarak_ban_kiri_dashed) + " cm", org_kiri_dashed, font, fontScale, color, thickness, cv2.LINE_AA)
            overlay_jarak_kiri_kanan = cv2.putText(overlay, "Jarak Ban Kanan Putus2 : " + str(jarak_ban_kanan_dashed) + " cm", org_kanan_dashed, font, fontScale, color, thickness, cv2.LINE_AA)
            overlay_jarak_kiri_kanan_kiri = cv2.putText(overlay, "Jarak Ban Kiri Sambung : " + str(jarak_ban_kiri_solid) + " cm",
                                                   org_kiri_solid, font, fontScale, color, thickness, cv2.LINE_AA)
            overlay_jarak_kiri_kanan_kiri_kanan = cv2.putText(overlay,
                                                        "Jarak Ban Kanan Sambung : " + str(jarak_ban_kanan_solid) + " cm",
                                                        org_kanan_solid, font, fontScale, color, thickness, cv2.LINE_AA)


            print("--------------------------------------------------------------------------" + str(jarak_ban_kiri_dashed))

            # cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (0, 420), (800, 420), (0, 0, 255), 3)
            # cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (0, 420), (300, 285), (0, 0, 255), 3)
            # cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (468, 285), (800, 420), (0, 0, 255), 3)
            # cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (468, 285), (300, 285), (0, 0, 255), 3)

            # cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (0, 1400), (2560, 1400), (0, 0, 255), 3)
            # cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (0, 1400), (1000, 950), (0, 0, 255), 3)
            # cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (1560, 950), (2560, 1400), (0, 0, 255), 3)
            # cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (1560, 950), (1000, 950), (0, 0, 255), 3)

            cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (0, 1300), (2559, 1300), (0, 0, 255), 3) #(_)
            cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (0, 1300), (1000, 1050), (0, 0, 255), 3) #(/)
            cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (1700, 1050), (2559, 1300), (0, 0, 255), 3) #(\)
            cv2.line(overlay_jarak_kiri_kanan_kiri_kanan, (1700, 1050), (1000, 1050), (0, 0, 255), 3) #(-)

            cv2.line(result_flipped_overlay, (200, 0), (200, 600), (0, 0, 255), 3)  # (-)

            cv2.circle(result_flipped_overlay, (275, 600), 5, (0, 0, 255), -1)
            cv2.circle(result_flipped_overlay, (190, 600), 5, (0, 0, 255), -1)

            # # jarak anatar kedua roda 750 pixel (2560x1600)
            # cv2.circle(frame, (1630, 1300), 5, (0, 0, 255), 3)
            # cv2.circle(frame, (880, 1300), 5, (0, 0, 255), 3)

            # jarak anatar kedua roda 750 pixel (800x480)
            # cv2.circle(result_flipped_ori, (285, 599), 10, (255, 0, 255), 3)
            # cv2.circle(result_flipped_ori, (200, 599), 10, (255, 0, 255), 3)


            ############FPS Calculate##########
            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = str(fps)
            print('FPS:', fps)

            cv2.imshow("LaneDetect",  overlay_jarak_kiri_kanan_kiri_kanan)
            cv2.imshow("MaskDetect", result_flipped_ori)
            cv2.imshow("InversePerspective", result_flipped_overlay)

            # save_path_mask = f"resultsVis/{date_time_str}.png"
            # cv2.imwrite(save_path_mask, overlay_jarak_kiri_kanan_kiri_kanan)
            # save_path_mask_IPM = f"resultsVis/{date_time_str}IPM.png"
            # cv2.imwrite(save_path_mask_IPM, result_flipped_overlay)


            # Plot each box with its label and score

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    streamer.release()
    cv2.destroyAllWindows()