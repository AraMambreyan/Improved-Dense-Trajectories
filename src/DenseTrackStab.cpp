#include "DenseTrackStab.h"
#include "Initialize.h"
#include "Descriptors.h"
#include "OpticalFlow.h"

#include <time.h>

#ifdef USE_PYTHON
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#endif

using namespace cv;
#ifdef USE_SURF
using namespace cv::xfeatures2d;
#endif

int show_track = 0; // set show_track = 1, if you want to visualize the trajectories

#ifdef USE_PYTHON
namespace {
	class ValidTrack
	{
	public:
		std::vector<Mat>::size_type frame_num;
		float mean_x;
		float mean_y;
		float var_x;
		float var_y;
		float length;
		float scale;
		float x_pos;
		float y_pos;
		float t_pos;
		std::vector<Point2f> coords;
		std::vector<Point2f> traj;
#ifdef USE_SURF
		std::vector<Point2f> displacement;
#endif
		std::vector<float> hog;
		std::vector<float> hof;
		std::vector<float> mbhX;
		std::vector<float> mbhY;

		ValidTrack() :
			frame_num(0), mean_x(0), mean_y(0), var_x(0), var_y(0),
			length(0), scale(0), x_pos(0), y_pos(0), t_pos(0) {
		}

		ValidTrack(std::vector<Mat>::size_type frame_num_,
			float mean_x_, float mean_y_, float var_x_, float var_y_,
			float length_, float scale_, float x_pos_, float y_pos_, float t_pos_,
			const std::vector<Point2f>& coords_, const std::vector<Point2f>& traj_,
			const std::vector<float>& hog_, const std::vector<float>& hof_,
			const std::vector<float>& mbhX_, const std::vector<float>& mbhY_) :
			frame_num(frame_num_), mean_x(mean_x_), mean_y(mean_y_),
			var_x(var_x_), var_y(var_y_), length(length_), scale(scale_),
			x_pos(x_pos_), y_pos(y_pos_), t_pos(t_pos_), coords(coords_), traj(traj_),
			hog(hog_), hof(hof_), mbhX(mbhX_), mbhY(mbhY_) {
		}

		PyObject* toPython() {
#ifdef USE_SURF
			if (!displacement.empty())
				return Py_BuildValue("(ifffffffffNNNNNNN)",
						frame_num, mean_x, mean_y, var_x, var_y, length,
						scale, x_pos, y_pos, t_pos,
						toPython(coords), toPython(traj), toPython(hog),
						toPython(hof), toPython(mbhX), toPython(mbhY), toPython(displacement));
#endif
			return Py_BuildValue("(ifffffffffNNNNNN)",
					frame_num, mean_x, mean_y, var_x, var_y, length,
					scale, x_pos, y_pos, t_pos,
					toPython(coords), toPython(traj), toPython(hog),
					toPython(hof), toPython(mbhX), toPython(mbhY));
		}

	private:
		PyObject* toPython(const std::vector<float>& values) {
			// Creating a Numpy array here provides no noticeable speedup and the
			// code would be longer.
			PyObject* py_list = PyList_New(values.size());
			for (size_t i = 0; i < values.size(); i++)
				PyList_SetItem(py_list, i, Py_BuildValue("f", values[i]));
			return py_list;
		}

		PyObject* toPython(const std::vector<Point2f>& values) {
			PyObject* py_list = PyList_New(values.size());
			for (size_t i = 0; i < values.size(); i++)
				PyList_SetItem(py_list, i, Py_BuildValue("[ff]", values[i].x, values[i].y));
			return py_list;
		}
	};

	int createDirectory(const char* path) {
		const char* end = strrchr(path, '/');
		if (end == NULL || end == path)
			return 0;
		size_t len = end - path;
		char dir_path[300];
		if (len + 1 > sizeof(dir_path)) {
			errno = ENAMETOOLONG;
			return -1;
		}
		memcpy(dir_path, path, len);
		dir_path[len] = '\0';
		for (char* p = dir_path + 1; *p; p++) {
			if (*p == '/') {
				*p = '\0';
				if (mkdir(dir_path, S_IRWXU) < 0 && errno != EEXIST)
					return -1;
				*p = '/';
			}
		}
		if (mkdir(dir_path, S_IRWXU) < 0 && errno != EEXIST)
			return -1;
		return 0;
	}
}
#endif

static
#ifdef USE_PYTHON
PyObject*
#else
void
#endif
densetrack(const std::vector<Mat>& video, int track_length, 
		int min_distance, int patch_size, int nxy_cell, int nt_cell, 
		int scale_num, int init_gap, int poly_n, double poly_sigma,
		const char* image_pattern, bool adjust_camera, float scale_stride) {
#ifdef USE_PYTHON
	std::vector<ValidTrack> valid_tracks;

	// Note that this opens a block closed by Py_END_ALLOW_THREADS.
	// https://docs.python.org/3/c-api/init.html#releasing-the-gil-from-extension-code
	// Variables may need to be declared outside that block.
	Py_BEGIN_ALLOW_THREADS
#endif

	TrackInfo trackInfo;
	DescInfo hogInfo, hofInfo, mbhInfo;

	InitTrackInfo(&trackInfo, track_length, init_gap);
	InitDescInfo(&hogInfo, 8, false, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&hofInfo, 9, true, patch_size, nxy_cell, nt_cell);
	InitDescInfo(&mbhInfo, 8, false, patch_size, nxy_cell, nt_cell);

	SeqInfo seqInfo;
	InitSeqInfo(&seqInfo, video);

#ifdef USE_SURF
	std::vector<Frame> bb_list;
	if(adjust_camera && bb_file) {
		LoadBoundBox(bb_file, bb_list);
		assert(bb_list.size() == seqInfo.length);
	}
#else
	adjust_camera = false;
#endif

//	fprintf(stderr, "video size, length: %d, width: %d, height: %d\n", seqInfo.length, seqInfo.width, seqInfo.height);

	if(show_track == 1)
		namedWindow("DenseTrackStab", 0);

#ifdef USE_SURF
	Ptr<SURF> detector_surf;
	Ptr<SURF> extractor_surf;
	if (adjust_camera) {
		detector_surf = SURF::create(200);
		extractor_surf = SURF::create(true, true);
	}

	std::vector<Point2f> prev_pts_flow, pts_flow;
	std::vector<Point2f> prev_pts_surf, pts_surf;
	std::vector<Point2f> prev_pts_all, pts_all;
	std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
	Mat prev_desc_surf, desc_surf, human_mask;
#endif

	Mat image, prev_grey, grey;

	std::vector<float> fscales(0);
	std::vector<Size> sizes(0);

	std::vector<Mat> prev_grey_pyr(0), grey_pyr(0), flow_pyr(0), flow_warp_pyr(0);
	std::vector<Mat> prev_poly_pyr(0), poly_pyr(0), poly_warp_pyr(0);

	std::vector<std::list<Track> > xyScaleTracks;
	int init_counter = 0; // indicate when to detect new feature points
#ifdef USE_PYTHON
	bool first_image = true;
#endif
	for (std::vector<Mat>::size_type frame_num = 0; frame_num != video.size(); frame_num++) {
		Mat frame = video[frame_num];
		if (frame.empty())
			break;
		if(frame_num == 0) {
			if (show_track == 1 || image_pattern != NULL)
				image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_grey.create(frame.size(), CV_8UC1);

			InitPry(frame, fscales, sizes, patch_size, scale_stride, scale_num);
			// for (size_t i = 0; i < fscales.size(); i++)
			// 	fprintf(stderr, "scale %lu: %.3f %d x %d\n", i, fscales[i], sizes[i].width, sizes[i].height);

			BuildPry(sizes, CV_8UC1, prev_grey_pyr);
			// for (size_t i = 0; i < prev_grey_pyr.size(); i++)
			// 	fprintf(stderr, "prev_grey %lu: %d x %d\n", i, prev_grey_pyr[i].cols, prev_grey_pyr[i].rows);
			BuildPry(sizes, CV_8UC1, grey_pyr);
			BuildPry(sizes, CV_32FC2, flow_pyr);

			BuildPry(sizes, CV_32FC(5), prev_poly_pyr);
			BuildPry(sizes, CV_32FC(5), poly_pyr);

#ifdef USE_SURF
			if (adjust_camera) {
				BuildPry(sizes, CV_32FC2, flow_warp_pyr);
				BuildPry(sizes, CV_32FC(5), poly_warp_pyr);
			}
#endif

			xyScaleTracks.resize(scale_num);

			frame.copyTo(prev_grey);
			if (show_track == 1 || image_pattern != NULL)
				cvtColor(frame, image, CV_GRAY2BGR);

			for(int iScale = 0; iScale < scale_num; iScale++) {
				if(iScale == 0)
					prev_grey.copyTo(prev_grey_pyr[0]);
				else {
					// fprintf(stderr, "resize %d: %d x %d -> %d x %d\n", iScale, prev_grey_pyr[iScale-1].size().width, prev_grey_pyr[iScale-1].size().height, prev_grey_pyr[iScale].size().width, prev_grey_pyr[iScale].size().height);
					resize(prev_grey_pyr[iScale-1], prev_grey_pyr[iScale], prev_grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);
				}

				// dense sampling feature points
				std::vector<Point2f> points(0);
				DenseSample(prev_grey_pyr[iScale], points, quality, min_distance);

				// save the feature points
				std::list<Track>& tracks = xyScaleTracks[iScale];
				for(unsigned int i = 0; i < points.size(); i++) {
					tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
#ifdef USE_SURF
					if (adjust_camera)
						tracks.back().disp.resize(trackInfo.length);
#endif
				}
			}

			// compute polynomial expansion
			my::FarnebackPolyExpPyr(prev_grey, prev_poly_pyr, fscales, poly_n, poly_sigma);

#ifdef USE_SURF
			if (adjust_camera) {
				human_mask = Mat::ones(frame.size(), CV_8UC1);
				if(bb_file)
					InitMaskWithBox(human_mask, bb_list[frame_num].BBs);

				detector_surf->detect(prev_grey, prev_kpts_surf, human_mask);
				extractor_surf->compute(prev_grey, prev_kpts_surf, prev_desc_surf);
			}
#endif

			continue;
		}

		init_counter++;
		frame.copyTo(grey);
		if (show_track == 1 || image_pattern != NULL)
			cvtColor(frame, image, CV_GRAY2BGR);

#ifdef USE_SURF
		if (adjust_camera) {
			// match surf features
			if(bb_file)
				InitMaskWithBox(human_mask, bb_list[frame_num].BBs);
			detector_surf->detect(grey, kpts_surf, human_mask);
			extractor_surf->compute(grey, kpts_surf, desc_surf);
			ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);
		}
#endif

		// compute optical flow for all scales once
		my::FarnebackPolyExpPyr(grey, poly_pyr, fscales, poly_n, poly_sigma);
		my::calcOpticalFlowFarneback(prev_poly_pyr, poly_pyr, flow_pyr, 10, 2, scale_stride);

#ifdef USE_SURF
		if (adjust_camera) {
			MatchFromFlow(prev_grey, flow_pyr[0], prev_pts_flow, pts_flow, human_mask);
			MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);

			Mat H = Mat::eye(3, 3, CV_64FC1);
			if(pts_all.size() > 50) {
				std::vector<unsigned char> match_mask;
				Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
				if(countNonZero(Mat(match_mask)) > 25)
					H = temp;
			}

			Mat H_inv = H.inv();
			Mat grey_warp = Mat::zeros(grey.size(), CV_8UC1);
			MyWarpPerspective(prev_grey, grey, grey_warp, H_inv); // warp the second frame

			// compute optical flow for all scales once
			my::FarnebackPolyExpPyr(grey_warp, poly_warp_pyr, fscales, poly_n, poly_sigma);
			my::calcOpticalFlowFarneback(prev_poly_pyr, poly_warp_pyr, flow_warp_pyr, 10, 2, scale_stride);
		}
#endif

		const std::vector<Mat>& fp =
#ifdef USE_SURF
			adjust_camera ? flow_warp_pyr :
#endif
			flow_pyr;
		for(int iScale = 0; iScale < scale_num; iScale++) {
			if(iScale == 0)
				grey.copyTo(grey_pyr[0]);
			else
				resize(grey_pyr[iScale-1], grey_pyr[iScale], grey_pyr[iScale].size(), 0, 0, INTER_LINEAR);

			int width = grey_pyr[iScale].cols;
			int height = grey_pyr[iScale].rows;

			// compute the integral histograms
			DescMat* hogMat = InitDescMat(height+1, width+1, hogInfo.nBins);
			HogComp(prev_grey_pyr[iScale], hogMat->desc, hogInfo);

			DescMat* hofMat = InitDescMat(height+1, width+1, hofInfo.nBins);
			HofComp(fp[iScale], hofMat->desc, hofInfo);

			DescMat* mbhMatX = InitDescMat(height+1, width+1, mbhInfo.nBins);
			DescMat* mbhMatY = InitDescMat(height+1, width+1, mbhInfo.nBins);
			MbhComp(fp[iScale], mbhMatX->desc, mbhMatY->desc, mbhInfo);

			// track feature points in each scale separately
			std::list<Track>& tracks = xyScaleTracks[iScale];
			for (std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end();) {
				int index = iTrack->index;
				Point2f prev_point = iTrack->point[index];
				int x = std::min<int>(std::max<int>(cvRound(prev_point.x), 0), width-1);
				int y = std::min<int>(std::max<int>(cvRound(prev_point.y), 0), height-1);

				Point2f point;
				point.x = prev_point.x + flow_pyr[iScale].ptr<float>(y)[2*x];
				point.y = prev_point.y + flow_pyr[iScale].ptr<float>(y)[2*x+1];
 
				if(point.x <= 0 || point.x >= width || point.y <= 0 || point.y >= height) {
					iTrack = tracks.erase(iTrack);
					continue;
				}

#ifdef USE_SURF
				if (adjust_camera) {
					iTrack->disp[index].x = flow_warp_pyr[iScale].ptr<float>(y)[2*x];
					iTrack->disp[index].y = flow_warp_pyr[iScale].ptr<float>(y)[2*x+1];
				}
#endif

				// get the descriptors for the feature point
				RectInfo rect;
				GetRect(prev_point, rect, width, height, hogInfo);
				GetDesc(hogMat, rect, hogInfo, iTrack->hog, index);
				GetDesc(hofMat, rect, hofInfo, iTrack->hof, index);
				GetDesc(mbhMatX, rect, mbhInfo, iTrack->mbhX, index);
				GetDesc(mbhMatY, rect, mbhInfo, iTrack->mbhY, index);
				iTrack->addPoint(point);

				// draw the trajectories at the first scale
				if((show_track == 1 || image_pattern != NULL) && iScale == 0)
					DrawTrack(iTrack->point, iTrack->index, fscales[iScale], image);

				// if the trajectory achieves the maximal length
				if(iTrack->index >= trackInfo.length) {
					std::vector<Point2f> trajectory(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory[i] = iTrack->point[i]*fscales[iScale];
				
#ifdef USE_SURF
					std::vector<Point2f> displacement;
					if (adjust_camera) {
						displacement.resize(trackInfo.length);
						for (int i = 0; i < trackInfo.length; ++i)
							displacement[i] = iTrack->disp[i]*fscales[iScale];
					}
#endif
	
					// Create a copy of the track coordinates because they are normalized by IsValid() call below.
					std::vector<Point2f> trajectory_copy(trackInfo.length+1);
					for(int i = 0; i <= trackInfo.length; ++i)
						trajectory_copy[i] = iTrack->point[i] * fscales[iScale];

					float mean_x(0), mean_y(0), var_x(0), var_y(0), length(0);
					if(IsValid(trajectory, mean_x, mean_y, var_x, var_y, length)
#ifdef USE_SURF
						 && (!adjust_camera || IsCameraMotion(displacement))
#endif
						 ) {
						// for spatio-temporal pyramid
						float x_pos = std::min<float>(std::max<float>(mean_x/float(seqInfo.width), 0), 0.999);
						float y_pos = std::min<float>(std::max<float>(mean_y/float(seqInfo.height), 0), 0.999);
						float t_pos = std::min<float>(std::max<float>((frame_num - trackInfo.length/2.0)/float(seqInfo.length), 0), 0.999);
#ifdef USE_PYTHON
						std::vector<float> hog;
						std::vector<float> hof;
						std::vector<float> mbhX;
						std::vector<float> mbhY;
						PrintDesc(iTrack->hog, hogInfo, trackInfo, hog);
						PrintDesc(iTrack->hof, hofInfo, trackInfo, hof);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo, mbhX);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo, mbhY);
						valid_tracks.push_back(ValidTrack(frame_num, mean_x, mean_y,
										var_x, var_y, length, fscales[iScale],
										x_pos, y_pos, t_pos, trajectory_copy, trajectory,
										hog, hof, mbhX, mbhY));
#ifdef USE_SURF
						if (!displacement.empty())
							// The vector is normalized. Make a copy if this isn't desired.
							valid_tracks.back().displacement = displacement;
#endif
#else
#ifndef USE_GPROF
						// output the trajectory
						printf("%lu\t%f\t%f\t%f\t%f\t%f\t%f\t", frame_num, mean_x, mean_y, var_x, var_y, length, fscales[iScale]);
						printf("%f\t", x_pos);
						printf("%f\t", y_pos);
						printf("%f\t", t_pos);
#ifdef USE_SURF
						if (adjust_camera) {
							// output the trajectory
							for (int i = 0; i < trackInfo.length; ++i)
								printf("%f\t%f\t", displacement[i].x, displacement[i].y);
						}
#endif
						PrintDesc(iTrack->hog, hogInfo, trackInfo);
						PrintDesc(iTrack->hof, hofInfo, trackInfo);
						PrintDesc(iTrack->mbhX, mbhInfo, trackInfo);
						PrintDesc(iTrack->mbhY, mbhInfo, trackInfo);
						printf("\n");
#endif
#endif
					}

					iTrack = tracks.erase(iTrack);
					continue;
				}
				++iTrack;
			}
			ReleDescMat(hogMat);
			ReleDescMat(hofMat);
			ReleDescMat(mbhMatX);
			ReleDescMat(mbhMatY);

			if(init_counter != trackInfo.gap)
				continue;

			// detect new feature points every gap frames
			std::vector<Point2f> points(0);
			for(std::list<Track>::iterator iTrack = tracks.begin(); iTrack != tracks.end(); iTrack++)
				points.push_back(iTrack->point[iTrack->index]);

			DenseSample(grey_pyr[iScale], points, quality, min_distance);
			// save the new feature points
			for(unsigned int i = 0; i < points.size(); i++) {
				tracks.push_back(Track(points[i], trackInfo, hogInfo, hofInfo, mbhInfo));
#ifdef USE_SURF
				if (adjust_camera)
					tracks.back().disp.resize(trackInfo.length);
#endif
			}
		}

		init_counter = 0;
		grey.copyTo(prev_grey);
		for(int i = 0; i < scale_num; i++) {
			grey_pyr[i].copyTo(prev_grey_pyr[i]);
			poly_pyr[i].copyTo(prev_poly_pyr[i]);
		}

#ifdef USE_SURF
		if (adjust_camera) {
			prev_kpts_surf = kpts_surf;
			desc_surf.copyTo(prev_desc_surf);
		}
#endif

		if( show_track == 1 ) {
			imshow( "DenseTrackStab", image);
			int c = cvWaitKey(3);
			if((char)c == 27) break;
		}
#ifdef USE_PYTHON
		if (image_pattern != NULL) {
			char path[300];
			snprintf(path, sizeof(path), image_pattern, frame_num);
			if (first_image) {
				first_image = false;
				if (createDirectory(path) < 0) {
					Py_BLOCK_THREADS
					return PyErr_SetFromErrno(PyExc_OSError);
				}
			}
			try {
				cv::imwrite(path, image);
			}
			catch (const cv::Exception& ex) {
				Py_BLOCK_THREADS
				PyErr_SetString(PyExc_RuntimeError, ex.what());
				return NULL;
			}
		}
#endif
	}

	if( show_track == 1 )
		destroyWindow("DenseTrackStab");

#ifdef USE_PYTHON
	Py_END_ALLOW_THREADS

	int cell_size = nxy_cell * nxy_cell * nt_cell;
	// PyList_Append increases the ref count (unlike PyList_SetItem)
	// https://stackoverflow.com/questions/3512414/does-this-pylist-appendlist-py-buildvalue-leak
	PyObject* dtype = PyList_New(adjust_camera ? 17 : 16);
	int idx = 0;
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "frame_num", "i", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "mean_x", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "mean_y", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "var_x", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "var_y", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "length", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "scale", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "x_pos", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "y_pos", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "t_pos", "f", 1));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, (i, i))", "coords", "f", track_length + 1, 2));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, (i, i))", "trajectory", "f", track_length, 2));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "hog", "f", 8 * cell_size));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "hof", "f", 9 * cell_size));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "mbh_x", "f", 8 * cell_size));
	PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, i)", "mbh_y", "f", 8 * cell_size));
#ifdef USE_SURF
	if (adjust_camera) {
		// Put displacement at the end to keep the other positions unchanged.
		PyList_SetItem(dtype, idx++, Py_BuildValue("(s, s, (i, i))", "displacement", "f", track_length, 2));
	}
#endif
	PyArray_Descr* descr;
	PyArray_DescrConverter(dtype, &descr);
	Py_DECREF(dtype);
	npy_intp dims[1];
	dims[0] = valid_tracks.size();
	PyArrayObject* py_tracks =
		(PyArrayObject*) PyArray_NewFromDescr(&PyArray_Type, descr, 1, dims, NULL, NULL, 0, NULL);
	if (!py_tracks) {
		fprintf(stderr, "Error creating Numpy array\n");
		return NULL;
	}

	npy_intp stride = PyArray_STRIDE(py_tracks, 0);
	char* bytes = PyArray_BYTES(py_tracks);
	for (size_t i = 0; i < valid_tracks.size(); i++) {
		PyObject* item = valid_tracks[i].toPython();
		PyArray_SETITEM(py_tracks, bytes + (stride * i), item);
		Py_DECREF(item);
		// Clear the track data
		ValidTrack tmp;
		std::swap(tmp, valid_tracks[i]);
	}

	return (PyObject*) py_tracks;
#endif
}

#ifdef USE_PYTHON
static PyObject*
densetrack_densetrack(PyObject* self, PyObject* args, PyObject* kwds) {
	static const char* arg_names[] = {
		"video", "track_length", "min_distance", "patch_size", "nxy_cell",
		"nt_cell", "scale_num", "init_gap", "poly_n", "poly_sigma",
		"image_pattern", "adjust_camera", "scale_stride", NULL};
	PyObject* video;
	int track_length = 15;
	int min_distance = 5;
	int patch_size = 32;
	int nxy_cell = 2;
	int nt_cell = 3;
	int scale_num = 8;
	int init_gap = 1;
	int poly_n = 7;
	int poly_sigma = 1.5;
	const char* image_pattern = NULL;
	int adjust_camera = 0;
	float scale_stride = sqrt(2);
	// https://docs.python.org/3.5/c-api/arg.html#other-objects
	// The object's reference count is not increased.
	if (!PyArg_ParseTupleAndKeywords
			(args, kwds, "O|iiiiiiiifspf", (char**)arg_names,
			&video, &track_length, &min_distance, &patch_size, &nxy_cell, &nt_cell,
			&scale_num, &init_gap, &poly_n, &poly_sigma, &image_pattern,
			&adjust_camera, &scale_stride))
		return NULL;
	// NPY_ARRAY_IN_ARRAY = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED
	PyObject* arr = PyArray_FROM_OTF(video, NPY_UBYTE, NPY_ARRAY_IN_ARRAY);
	if (arr == NULL)
		return NULL;
	if (PyArray_NDIM((PyArrayObject*)arr) != 3) {
		PyErr_SetString(PyExc_ValueError, 
			"'video' has to have 3 dimensions (frames, height, width)");
		Py_DECREF(arr);
		return NULL;
	}
	unsigned char* data = (unsigned char*)PyArray_DATA((PyArrayObject*)arr);
	npy_intp len = PyArray_DIM((PyArrayObject*)arr, 0);
	npy_intp rows = PyArray_DIM((PyArrayObject*)arr, 1);
	npy_intp cols = PyArray_DIM((PyArrayObject*)arr, 2);
	if (std::min(rows, cols) < patch_size * sqrt(2)) {
		PyErr_SetString(PyExc_ValueError,
			"min dimension has to be at least patch_size*sqrt(2)");
		Py_DECREF(arr);
		return NULL;
	}
	std::vector<Mat> frames;
	frames.reserve(len);
	for (int i = 0; i < len; i++) {
		Mat frame = Mat(rows, cols, CV_8UC1, (data + i*rows*cols));
		frames.push_back(frame);
	}
	// Parallel OpenCV isn't faster here but uses more CPU.
	setNumThreads(0);
	PyObject* result = densetrack(frames, track_length, min_distance, patch_size, nxy_cell,
		nt_cell, scale_num, init_gap, poly_n, poly_sigma,
		image_pattern, adjust_camera, scale_stride);
	Py_DECREF(arr);
	return result;
}

static PyMethodDef DenseTrackMethods[] = {
	{"densetrack", (PyCFunction)densetrack_densetrack, METH_VARARGS | METH_KEYWORDS,
	 "Computes dense trajectories for a video."},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef densetrackmodule = {
	PyModuleDef_HEAD_INIT,
	"densetrack",
	NULL,
	-1,
	DenseTrackMethods
};

static PyObject* DenseTrackError;

PyMODINIT_FUNC
PyInit_densetrack(void)
{
	PyObject* m = PyModule_Create(&densetrackmodule);
	if (m == NULL)
		return NULL;
	DenseTrackError = PyErr_NewException("densetrack.error", NULL, NULL);
	Py_INCREF(DenseTrackError);
	PyModule_AddObject(m, "error", DenseTrackError);
	import_array();
	return m;
}
#endif

#ifndef USE_PYTHON
int main(int argc, char** argv)
{
	VideoCapture capture;
	char* video = argv[1];
	arg_parse(argc, argv);
	capture.open(video);
	if (!capture.isOpened()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}
	std::vector<Mat> frames;
	for (int frame_num = 0; frame_num <= end_frame; frame_num++) {
		Mat frame;
		// get a new frame
		capture >> frame;
		if (frame.empty())
			break;
		if (frame_num >= start_frame) {
			Mat gray;
			cvtColor(frame, gray, CV_BGR2GRAY);
			frames.push_back(gray);
		}
	}
	if (frames.empty()) {
		fprintf(stderr, "Could not initialize capturing..\n");
		return -1;
	}
	densetrack(frames, track_length, min_distance,
		 patch_size, nxy_cell, nt_cell, scale_num, init_gap, 7, 1.5,
		 NULL, true, sqrt(2));
	return 0;
}
#endif
