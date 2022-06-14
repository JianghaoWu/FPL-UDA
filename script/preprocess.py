import os 
import SimpleITK as sitk 
import numpy as np 

def get_image_info():
    img_dir = "/home/guotai/disk2t/data/VS/CrossMoDA/crossmoda_training/source_training"
    img_names = os.listdir(img_dir)
    lab_names = [item for item in img_names if "Label.nii.gz" in item]
    lab_names = sorted(lab_names)
    num = len(lab_names)
    print("image number", num)

    dmin_list, dmax_list = [], []
    hmin_list, hmax_list = [], []
    wmin_list, wmax_list = [], []
    for lab_name in lab_names:
        lab_obj = sitk.ReadImage(img_dir + '/' + lab_name)
        lab = sitk.GetArrayFromImage(lab_obj)
        spacing = lab_obj.GetSpacing()
        [D, H, W] = lab.shape 
        # print(lab_name, D, H, W)

        # get upper and lower bound of labels
        indices = np.where(lab>0)
        d0, d1 = D - indices[0].max(), D - indices[0].min()
        dp0, dp1 = d0 * spacing[2], d1 * spacing[2]
        dmin_list.append(dp0)
        dmax_list.append(dp1)

        h0, h1 = indices[1].min(), indices[1].max()
        w0, w1 = indices[2].min(), indices[2].max()
        hmin_list.append(h0)
        hmax_list.append(h1)
        wmin_list.append(w0)
        wmax_list.append(w1)
        print(lab_name, D, H, W, d0, d1, dp0, dp1)
    dmin_list, dmax_list = np.asarray(dmin_list), np.asarray(dmax_list)
    hmin_list, hmax_list = np.asarray(hmin_list), np.asarray(hmax_list)
    wmin_list, wmax_list = np.asarray(wmin_list), np.asarray(wmax_list)
    print("upper bound, min, mean, max", dmin_list.min(), dmin_list.mean(), dmin_list.max())
    print("lower bound, min, mean, max", dmax_list.min(), dmax_list.mean(), dmax_list.max())
    print("inferior bound,  min, mean, max", hmin_list.min(), hmin_list.mean(), hmin_list.max())
    print("posterior bound, min, mean, max", hmax_list.min(), hmax_list.mean(), hmax_list.max())
    print("left bound,  min, mean, max", wmin_list.min(), wmin_list.mean(), wmin_list.max())
    print("right bound, min, mean, max", wmax_list.min(), wmax_list.mean(), wmax_list.max())


def image_crop():
    img_dir = "/home/guotai/disk2t/data/VS/CrossMoDA/crossmoda_training/source_training"
    out_dir = "/home/guotai/disk2t/data/VS/CrossMoDA/crossmoda_training/source_training_crop"
    img_names = os.listdir(img_dir)
    img_names = [item for item in img_names if "ceT1" in item]
    for img_name in img_names:
        lab_name = img_name.replace("ceT1", "Label")
        img_obj = sitk.ReadImage(img_dir + '/' + img_name)
        lab_obj = sitk.ReadImage(img_dir + '/' + lab_name)
        img = sitk.GetArrayFromImage(img_obj)
        lab = sitk.GetArrayFromImage(lab_obj)

        [D, H, W] = img.shape
        spacing = img_obj.GetSpacing()
        # crop image based on predefined bounding box
        d0 = int(D - 155/spacing[2])
        d1 = int(D - 95/spacing[2])
        h0, h1 = 205, 365
        w0, w1 = 120, 392
        img_sub = img[d0:d1, h0:h1, w0:w1]
        lab_sub = lab[d0:d1, h0:h1, w0:w1]
        assert(lab_sub.sum() == lab.sum())
            # print(img_name, lab_sub.sum(), lab.sum())

        #convert array to image object
        out_img_obj = sitk.GetImageFromArray(img_sub)
        out_lab_obj = sitk.GetImageFromArray(lab_sub)
        direct = img_obj.GetDirection()
        origin = img_obj.GetOrigin()
        out_img_obj.SetSpacing(spacing)
        out_img_obj.SetDirection(direct)
        out_img_obj.SetOrigin(origin)
        sitk.WriteImage(out_img_obj, out_dir + '/' + img_name)

        out_lab_obj.CopyInformation(out_img_obj)
        sitk.WriteImage(out_lab_obj, out_dir + '/' + lab_name)

if __name__ == "__main__":
    # get_image_info()
    image_crop()