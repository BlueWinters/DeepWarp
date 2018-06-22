import cv2
import numpy as np
import json
from glob import glob

json_fns = ['all/0001/1.json']#glob("imgs/*.json")

for json_fn in json_fns:

	img = cv2.imread("%s.jpg"%json_fn[:-5])
	data_file = open(json_fn)
	data = json.load(data_file)

	def process_json_list(json_list):
		ldmks = [eval(s) for s in json_list]
		return np.array([(x, img.shape[0]-y, z) for (x,y,z) in ldmks])

	ldmks_interior_margin = process_json_list( data['interior_margin_2d'])
	ldmks_caruncle = process_json_list( data['caruncle_2d'])
	ldmks_iris = process_json_list( data['iris_2d'])

	# Draw black background points and lines
	for ldmk in np.vstack([ldmks_interior_margin, ldmks_caruncle, ldmks_iris[::2]]):
		cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 3, (0,0,0),-1)
	# cv2.polylines(img, np.array([ldmks_interior_margin[:,:2]], int), True, (0,0,0), 2)
	# cv2.polylines(img, np.array([ldmks_iris[:,:2]], int), True, (0,0,0), 2)

	# Draw green foreground points and lines
	for n, ldmk in enumerate(np.vstack([ldmks_interior_margin, ldmks_caruncle, ldmks_iris[::2]])):
		if n in [22, 2, 5, 8, 11, 14]:
			cv2.putText(img, str(n), (int(ldmk[0]), int(ldmk[1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
			cv2.circle(img, (int(ldmk[0]), int(ldmk[1])), 2, (0,255,0),-1)
	# cv2.polylines(img, np.array([ldmks_interior_margin[:,:2]], int), True, (0,255,0), 1)
	# cv2.polylines(img, np.array([ldmks_iris[:,:2]], int), True, (0,255,0), 1)

	look_vec = list(eval(data['eye_details']['look_vec']))

	eye_c = np.mean(ldmks_iris[:,:2], axis=0).astype(int)
	look_vec[1] = -look_vec[1]
	cv2.line(img, tuple(eye_c), tuple(eye_c+(np.array(look_vec[:2])*80).astype(int)), (0,0,0), 3)
	cv2.line(img, tuple(eye_c), tuple(eye_c+(np.array(look_vec[:2])*80).astype(int)), (0,255,255), 2)

	cv2.imshow("syntheseyes_img", img)
	# cv2.imwrite("imgs/annotated_%s.png".format(json_fn[:-5]), img)
	cv2.waitKey(0)