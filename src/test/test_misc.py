
# @pytest.mark.skip(reason="Not ready")
# def test_phase():
#   shifts = 12
#   stripe_counts = np.array([1.0, 8.0, 64.0])
#   N = shifts * len(stripe_counts)

#   exp_path = Path(filedialog.askdirectory(title="Where is the folder for the experiment?"))
#   img_repo = FileImageRepo(exp_path / "images", file_ext='.tif', greyscale=True)

#   camera = devices.FileCamera(resolution=(1080, 1920), channels=1,
#     imgs=[img_repo.get("calibration" + str(i+1).zfill(5)) for i in range(N)])

#   projector = devices.FakeProjector(resolution=(1080, 1920))

#   shifter = phase.NStepPhaseShift(shifts)
#   unwrapper = phase.MultiFreqPhaseUnwrap(stripe_counts)
  
#   calibrator = profilometry.StereoCalibrator(cb_size=(10, 7), square_width=0.018, shift_mask=0.1)
#   phasemap = calibrator.gather_phasemap(camera, projector, shifter, unwrapper)

# @pytest.mark.skip(reason="Not ready")
# def test_cv2camera():
#   # camera = devices.CV2Camera(0, resolution=(720, 1280), channels=1)

#   # camera.show_feed()

#   devices_path = Path(filedialog.askdirectory(title="What directory for loading devices?"))
#   cam_repo = FileCameraRepo(devices_path, overwrite=True)

#   camera = cam_repo.get("camera1")

#   camera.show_feed()

# @pytest.mark.skip(reason="Not ready")
# def test_projector():

#   projector = devices.DisplayProjector()

#   devices_path = Path(filedialog.askdirectory(title="What directory for loading devices?"))
#   proj_repo = FileProjectorRepo(devices_path, overwrite=True)

#   proj_repo.add(projector, "projector1")

# @pytest.mark.skip(reason="Not ready")
# def test_gamma():
#   global ex_service
#   imgs = list(ex_service.get_by(f"gamma*", sorted=True))

#   measured_gammas = np.array([image.calc_gamma(img.raw_data) for img in imgs])
#   expected_gammas = np.linspace(0.0, 1.0, len(measured_gammas), dtype=np.float32)

#   image.show_scatter([expected_gammas], [measured_gammas])

# @pytest.mark.skip(reason="Not ready")
# def test_vignette():
#   global ex_service
#   img = ex_service.load_img("gamma1").raw_data

#   diff = (np.ones_like(img, dtype=np.float32) * img.max()) - img
  
#   image.show_image(diff, "Vignetting amount")
#   image.show_image(img + diff, "Vignetting fixed")

# @pytest.mark.skip(reason="Not ready")
# def test_curve():
#   global ex_service
#   imgs = list(ex_service.get_by("intensity", sorted=True))

#   h, w = imgs[0].raw_data.shape

#   d = 25

#   h1 = int(h / 2) - d
#   h2 = h1 + 2 * d

#   w1 = int(w / 2)
#   w2 = w1 + 2 * d

#   intensities = [np.mean(img.raw_data[h1:h2, w1:w2]) for img in imgs]
#   wattages = np.linspace(30.0, 36.625, len(imgs), endpoint=True)

#   plt.ylim(0.0, 1.01)
#   plt.scatter(wattages, intensities, c='r')
#   plt.show()

#   found = next((v for (i, v) in enumerate(wattages[:-1]) if (intensities[i] == intensities[i+1] == 1.0)), None)

#   if not found:
#     print("No max intensity was found given the range of wattages")
#     return True
  
#   print(f"Max intensity wattage: {found}")
