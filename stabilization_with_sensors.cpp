//===================================================================
//=              ** Drone Projected Image Stabilization             =
//= Programmer    : Eunbin Choi                                     =
//= Date          : 2019-11-10                                      =
//= Updates       : MODEL CHANGING                                  =                                  
//===================================================================

#include <iostream>
#include <algorithm>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <string>
#include <stdlib.h>   
#include <math.h>
#include <fcntl.h>   // File Control Definitions
#include <termios.h> // POSIX Terminal Control Definitions
#include <unistd.h>  // UNIX Standard Definitions
#include <errno.h>   // ERROR Number Definitions
#include <time.h>
#include <limits>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <atomic>
#include <cstdlib>
#include <pthread.h>
#include <queue>
#include <chrono>
#include <mutex>
#include <vector>
#include <cstdlib>
#include <X11/Xlib.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#define PI 3.14159265
#define READ_BUF_SIZE_ALTITUDE 10
#define READ_BUF_SIZE_DISTANCE 10
#define READ_BUF_SIZE_AHRS 48
#define UART_PORT_AHRS "/dev/ttyTHS1"
#define UART_PORT_DISTANCE "/dev/ttyDIS1"
#define UART_PORT_ALTITUDE "/dev/ttyALT1"
#define PRECISION 100

#define DISPLAY_LOG 1

#if DISPLAY_LOG == 0
#define printf(...) ;
#endif


// resolution 
#define DISPLAY_RES_WIDTH 640
#define DISPLAY_RES_HEIGHT 360
#define MEDIA_RES_WIDTH 213
#define MEDIA_RES_HEIGHT 120
#define INFO_RES_WIDTH 640
#define INFO_RES_HEIGHT 120

#define FRAME_ROI_TOP  120
#define FRAME_ROI_BOTTOM 240
#define FRAME_ROI_LEFT  213
#define FRAME_ROI_RIGHT 427


//#define DISPLAY_RES_WIDTH 960
//#define DISPLAY_RES_HEIGHT 540

//#define MEDIA_RES_WIDTH 319
//#define MEDIA_RES_HEIGHT 180
//#define INFO_RES_WIDTH 960
//#define INFO_RES_HEIGHT 180

//#define FRAME_ROI_TOP  180
//#define FRAME_ROI_BOTTOM 360
//#define FRAME_ROI_LEFT  319
//#define FRAME_ROI_RIGHT 640



#define FRAME_ROI_OUTSIDE_TOP 0
#define FRAME_ROI_OUTSIDE_LEFT 0

#define proj_angle          24.5
#define proj_angle_rad      0.4
#define proj_angle_rad_half 0.2
#define angle_threshold     3

using namespace cv;
using namespace std;
typedef enum {
	SENSORS_IDX_ROLL = 0,
	SENSORS_IDX_PITCH = 1,
	SENSORS_IDX_YAW = 2,
	SENSORS_IDX_DISTANCE = 3,
	SENSORS_IDX_ALTITUDE = 4

} SENSORS_IDX;

typedef enum {
	PLAYMODE_IMAGE = 0,
	PLAYMODE_VIDEO = 1

} PLAYMODE;

typedef struct {
	char filename[1024];
	int  mode;

} MEDIAINFO;

// Global Variable 
char   dirbuff[1024];  
char   path_buff[1024];
string path;

char foldername_str[1024];
char filename_buff[1024];
char filename[9][1024] = {"SENSORS_VALUE","MATRIX_VALUE", "ROTATION_SENSORS_GAIN_TIME", \
	"DISTANCE_SENSORS_GAIN_TIME", "ALTITUDE_SENSORS_GAIN_TIME", "PROJECTION_TIME", "MATRIX_CALCULATION_TIME", \
		"FRAME_PER_SEC", "TOTAL_TIME(FPS)"};

clock_t start_rotation_sensors_time;   clock_t end_rotation_sensors_time;
clock_t start_distance_sensors_time;   clock_t end_distance_sensors_time;
clock_t start_altitude_sensors_time;   clock_t end_altitude_sensors_time;
clock_t start_projection_time;         clock_t end_projection_time;
clock_t start_matrix_calculation_time; clock_t end_matrix_calculation_time;
clock_t start_frame_per_sec;           clock_t end_frame_per_sec;
clock_t start_total_time;              clock_t end_total_time;

int file_res = 0;

MEDIAINFO media_list[] = { {"flower_img_only_HD.png",   PLAYMODE_IMAGE}
	, {"checker_img_only_HD.png",  PLAYMODE_IMAGE} \
		, {"measure1_img_only_HD.png", PLAYMODE_IMAGE} \
		, {"measure2_img_only_HD.png", PLAYMODE_IMAGE} \
		, {"videoplayback.mp4",        PLAYMODE_VIDEO} 
	, {"blackimage.png",           PLAYMODE_IMAGE}};

time_t current_time = time(NULL);

char    buffertmp[1024];
char    file_buffer[1024];
struct tm* struct_time = localtime(&current_time);
bool g_bRunThreadAHRS     = true;
bool g_bRunThreadAltitude = true;
bool g_bRunThreadDistance = true;
//bool g_bRunThreaddistance   = true;

// Shared Variables

float sensors[5] = {0.0, 0.0, 0.0, 1.0, 0.0}; // Roll, Pitch, Yaw, Distance, Altitude, distance (distance);
float CMtoPIXEL;

// Function Init.
void delay (clock_t n);
void setting_serialport(int fd, speed_t baudrate, int wait_ch);

static int lookup(const char* arg);
char* file_create(char* filename);
void* threadforAHRS(void*);
void* threadforDistance(void*);
void* threadforAltitude(void*);
void transform_ROI( const Mat& src, Mat& dst, const Mat& M0, Size dsize, int flags, int borderType, const Scalar& borderValue, Point origin );
void transform_ori( const Mat& src, Mat& dst, const Mat& M0, Size dsize, int flags, int borderType, const Scalar& borderValue, Point origin );

//void* threadfordistance(void*);

bool  is_roll_initialized = false;
bool  is_pitch_initialized = false;
bool  is_yaw_initialized = false;
bool  is_distance_initialized = false;
bool  is_altitude_initialized = false;

float initial_roll;
float initial_pitch;
float initial_yaw;
float initial_distance;
float initial_altitude;

float prev_real_roll;
float prev_real_pitch;
float prev_real_yaw;
float prev_real_altitude;
float prev_real_distance;
//float prev_real_distance;

int cnt_for_distance = 0;
int cnt_for_altitude = 0;
//int cnt_for_distance = 0;

bool  stabilization_rotation_on = true;
bool  stabilization_distance_on = true;
bool  stabilization_altitude_on = true;
//bool  stabilization_distance_on = true;

Mat pers_proj_roll;	
Mat pers_proj_pitch;
Mat pers_proj_yaw;
Mat pers_proj_roll_pitch_yaw_distance_altitude;
Mat H_matrix;
int frame_cnt = 0;

int main( int argc, char** argv ) {

	int thresult, status, i; 
	vector<void *(*)(void *)> thread_list;
	vector<pthread_t> tident(10);

	thread_list.push_back(threadforAHRS);
	thread_list.push_back(threadforAltitude);
	thread_list.push_back(threadforDistance);

	for (i = 0; i < thread_list.size(); i++) {

		if (pthread_create (&tident[i], NULL, thread_list[i], (void*)NULL) < 0){
			perror ("error:");
			exit(0);
		}
	}

	Mat frame(DISPLAY_RES_HEIGHT, DISPLAY_RES_WIDTH, CV_8UC3);
	//Mat frame_ROI(FRAME_ROI_LEFT, FRAME_ROI_TOP, CV_8UC3);

	//initial media
	Mat image_input = imread(media_list[0].filename, cv::IMREAD_COLOR);
	int media_mode = media_list[0].mode;

	Mat black_image = imread(media_list[5].filename, cv::IMREAD_COLOR);

	VideoCapture video_cap;
	Mat video_input;

	// resized media for putting into the frame
	Mat resized_media;
	//Mat resized_blackimage;

	// frame after warping
	Mat frame_warped;

	float Roll = 0, Pitch = 0, Yaw = 0;//, dist_x = 0, dist_y = 0, dist_z = 0;
	//int Roll_sign = 1, Pitch_sign = 1, Yaw_sign = 1;

	Size frame_res(DISPLAY_RES_WIDTH, DISPLAY_RES_HEIGHT); // image size(spatial resolution)
	Size media_res(MEDIA_RES_WIDTH, MEDIA_RES_HEIGHT); // image size(spatial resolution) 
	//Size black_res(INFO_RES_WIDTH, INFO_RES_HEIGHT); // image size(spatial resolution)

	//Size frame_res(1280, 720); // image size(spatial resolution)
	Point3f pt(0, DISPLAY_RES_WIDTH >> 1, DISPLAY_RES_HEIGHT >> 1); //translation offset
	//Point3f pt(0, 640, 360); //translation offset

	resize(image_input, resized_media, media_res); 
	//resize(black_image, resized_blackimage, black_res);
	resized_media.copyTo(frame(Rect(FRAME_ROI_LEFT, FRAME_ROI_TOP, MEDIA_RES_WIDTH, MEDIA_RES_HEIGHT)));
	//resized_blackimage.copyTo(frame(Rect(FRAME_ROI_OUTSIDE_LEFT, FRAME_ROI_OUTSIDE_TOP, INFO_RES_WIDTH,INFO_RES_HEIGHT)));

	namedWindow ("FRAME", cv::WINDOW_NORMAL);  
	cv::setWindowProperty("FRAME", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);

	float Roll_r, Pitch_r, Yaw_r;
	float f = 1, altitude, distance; 
	float trans_by_distance;

	// Text overlay on the frame
	Point textPos    (FRAME_ROI_LEFT + 10, FRAME_ROI_TOP + 30);
	// Point sensorPos  (FRAME_ROI_OUTSIDE_LEFT + 50, FRAME_ROI_OUTSIDE_TOP + 100);
	Point sensorPos  (FRAME_ROI_OUTSIDE_LEFT + 30, FRAME_ROI_OUTSIDE_TOP  + 100);
	Point matrixPos1 (FRAME_ROI_OUTSIDE_LEFT + 50 , FRAME_ROI_OUTSIDE_TOP + 260);
	Point matrixPos2 (FRAME_ROI_OUTSIDE_LEFT + 50 , FRAME_ROI_OUTSIDE_TOP + 290);
	Point matrixPos3 (FRAME_ROI_OUTSIDE_LEFT + 50 , FRAME_ROI_OUTSIDE_TOP + 320);

	int textFace = FONT_HERSHEY_SIMPLEX;
	double textScale = 1;
	double textThickness = 2;

	char text_buffer[16] = "";
	char sensor_buffer[50] ="";

	char* buffer0 = file_create (filename[0]); // -1: abnormal, 0: normal, 1: already exist
	fstream outfile_sensors(buffer0,std::fstream::app | std::fstream::out);	

	char* buffer1 = file_create (filename[1]);
	fstream outfile_matrix (buffer1, std::fstream::out | std::fstream::app) ;

	char* buffer6 = file_create (filename[6]);
	fstream outfile_matrix_calculation_gain_time (buffer6, std::fstream::out | std::fstream::app);	

	char* buffer7 = file_create (filename[5]);	
	fstream outfile_projection_gain_time(buffer7, std::fstream::out | std::fstream::app);


	//char* buffer8 = file_create (filename[7]);	
	//fstream outfile_frame_per_sec(buffer8, std::fstream::out | std::fstream::app);

	char* buffer9 = file_create (filename[8]);	
	fstream outfile_total_time(buffer9, std::fstream::out | std::fstream::app);

	while(true) {

		start_total_time = clock();


		if (media_mode == PLAYMODE_VIDEO) {
			// read one video frame
			video_cap >> video_input;

			Rect blankRoi (0, 0, DISPLAY_RES_WIDTH, DISPLAY_RES_HEIGHT);
			frame (blankRoi).setTo(Scalar(0));


			// resize video frame to display
			resize(video_input, resized_media, media_res); 

			// copy video to frame ROI
			resized_media.copyTo(frame(Rect(FRAME_ROI_LEFT, FRAME_ROI_TOP, MEDIA_RES_WIDTH, MEDIA_RES_HEIGHT)));

			sprintf(text_buffer, "%d", frame_cnt);

			putText(frame, text_buffer, textPos, textFace, textScale, Scalar(0, 0, 0), textThickness);

			//  sprintf(sensor_buffer, "R:%4.2f, P:%4.2f, Y:%4.2f", sensors[0], sensors[1], sensors[2]);            
			//  sprintf(sensor_buffer, "R:%4.2f, P:%4.2f, Y:%4.2f, dist: %4.2f", sensors[0], sensors[1], sensors[2], sensors[3]);
			sprintf(sensor_buffer, "R:%4.2f, P:%4.2f, Y:%4.2f, dist: %4.2f, alti: %4.2f", sensors[SENSORS_IDX_ROLL], sensors[SENSORS_IDX_PITCH], sensors[SENSORS_IDX_YAW], sensors[SENSORS_IDX_DISTANCE], sensors[SENSORS_IDX_ALTITUDE]);

			putText(frame, sensor_buffer, sensorPos, textFace, textScale * 0.7, Scalar(255, 255,255), textThickness );

			sprintf(sensor_buffer, "H: %4.2f, %4.2f, %4.2f", H_matrix.at<float>(0, 0), H_matrix.at<float>(0, 1), H_matrix.at<float>(0, 2));

			putText(frame, sensor_buffer, matrixPos1, textFace, textScale * 0.7, Scalar(255, 255,255), textThickness );

			sprintf(sensor_buffer, "   %4.2f, %4.2f, %4.2f", H_matrix.at<float>(1, 0), H_matrix.at<float>(1, 1), H_matrix.at<float>(1, 2));

			putText(frame, sensor_buffer, matrixPos2, textFace, textScale * 0.7, Scalar(255, 255,255), textThickness );

			sprintf(sensor_buffer, "   %4.2f, %4.2f, %4.2f", H_matrix.at<float>(2, 0), H_matrix.at<float>(2, 1), H_matrix.at<float>(2, 2));

			putText(frame, sensor_buffer, matrixPos3, textFace, textScale * 0.7, Scalar(255, 255,255), textThickness );

			//start_frame_per_sec = clock();
			frame_cnt++;

		}

		start_matrix_calculation_time = clock();

		for (int h = 0 ; h < 5 ; h++){
			sensors[h] = ceil(sensors[h] * 10)/10; 
		}

		if (stabilization_rotation_on){
			Roll = sensors[SENSORS_IDX_ROLL];
			Pitch = sensors[SENSORS_IDX_PITCH];;
			Yaw = sensors[SENSORS_IDX_YAW];

		}
		else {
			Roll  = initial_roll;;
			Pitch = initial_pitch;;
			Yaw   = initial_yaw;;
		}

		Roll_r  = Roll * PI / 180.0;
		Pitch_r = Pitch * PI / 180.0 * 0.0001;
		Yaw_r   = Yaw * PI / 180.0* 0.0001;


		if (stabilization_distance_on) {
			distance = sensors[SENSORS_IDX_DISTANCE];
			//printf("   \n @@ ditance : changed f :%f\n", sensors[3]);


		} 
		else {
			distance = initial_distance;

		}

		if (stabilization_altitude_on){
			altitude = sensors[SENSORS_IDX_ALTITUDE];
			//altitude = altitude * CMtoPIXEL;

			//printf("   \n ## altitude : changed f :%f\n", altitude);
		}
		else {
			altitude = initial_altitude;

		}

		CMtoPIXEL = 360.0 / (distance * tan(proj_angle_rad));
		trans_by_distance = (tan(proj_angle_rad)/2.0) * (distance - initial_distance /*- distance*/) * CMtoPIXEL;

		altitude = altitude * CMtoPIXEL;

		//printf("**MAIN sensors: %f %f %f %f\n", Roll_r, Pitch_r, Yaw_r, f);
		/*printf("**MAIN SENSORS: roll: %f pitch: %f yaw: %f dist: %f alti: %f\n", sensors[SENSORS_IDX_ROLL]
		  , sensors[SENSORS_IDX_PITCH]
		  , sensors[SENSORS_IDX_YAW]
		  , sensors[SENSORS_IDX_DISTANCE]
		  , sensors[SENSORS_IDX_ALTITUDE]
		  );

*/

		Point3f trOffset(0, DISPLAY_RES_WIDTH/2, DISPLAY_RES_HEIGHT/2);
		float data_translate[9] = {1, 0, trOffset.y, 0, 1, trOffset.z, 0, 0, 1};
		float data_translate_minus[9] = {1, 0, -trOffset.y, 0, 1, -trOffset.z, 0, 0, 1};

		Mat  H_translate               = Mat(3, 3, CV_32F, data_translate);
		Mat  H_translate_minus         = Mat(3, 3, CV_32F, data_translate_minus);

		Point3f distance_point (0, 0, trans_by_distance);
		Point3f altitude_point (0, 0, altitude);

		float data_translate_distance[9]\
			= {(float)initial_distance/distance,0,0,0,(float)initial_distance/distance, distance_point.z, 0,0,1};
		Mat H_translate_distance         = Mat(3, 3, CV_32F, data_translate_distance);

		float data_translate_altitude[9] = {1,0,0, 0,1,altitude_point.z, 0,0,1};	
		Mat  H_translate_altitude        = Mat(3, 3, CV_32F, data_translate_altitude);


		//start2 = clock();
		float distance_used = distance * CMtoPIXEL;

		float roll_trans_tx = distance_used*tan(proj_angle_rad_half)*sin(Roll_r);
		float roll_trans_ty = distance_used*tan(proj_angle_rad_half)*(1-cos(Roll_r));
		float Pitch_r_mul = Pitch_r * 10000;
		float Yaw_r_mul = Yaw_r * 10000;

		//f = 1; 
		// compensate matrix (not defined distortion) // stabilization
		float data_roll[9] =\
		{f*cos(Roll_r), f*sin(Roll_r), -roll_trans_tx ,\
			-f*sin(Roll_r), f*cos(Roll_r), -roll_trans_ty,\
				0, 0, 1};     

		float pitch_trans_ori = (distance_used) / (cos(proj_angle_rad_half));
		float pitch_trans_tilt = (distance_used)  / (cos((proj_angle_rad_half)-Pitch_r_mul));
		float pitch_trans;

		if (Pitch_r > 0) {pitch_trans = sqrtl(powl(pitch_trans_tilt*sin(Pitch_r_mul),2) + powl(pitch_trans_ori-pitch_trans_tilt*cos(Pitch_r_mul),2));}
		else {pitch_trans = -(sqrtl(powl(pitch_trans_tilt*sin(Pitch_r_mul),2) + powl(pitch_trans_ori-pitch_trans_tilt*cos(Pitch_r_mul),2)));}


		float data_pitch[9] = \
		{f,0/*-pt.y*sin(Pitch_r)*/, 0, \
			0, f*cos(Pitch_r)/*-pt.z*sin(Pitch_r)*/,pitch_trans,/* CMtoPIXEL*(((distance) / (cos(proj_angle_rad/2.0) + Pitch_r))  - ((distance) / cos(proj_angle_rad/2.0)))*/\
				0, -sin(Pitch_r), 1};


		float data_yaw[9]   =\
		{f*cos(Yaw_r)/*+pt.y*sin(Yaw_r)*/, 0, -(distance) * CMtoPIXEL * tan(Yaw_r_mul),\
			0,/*+pt.z*sin(Yaw_r)*/ f, 0, \
				+sin(Yaw_r), 0, 1};	



		pers_proj_roll            = Mat(3, 3, CV_32F, data_roll);
		pers_proj_pitch           = Mat(3, 3, CV_32F, data_pitch);
		pers_proj_yaw             = Mat(3, 3, CV_32F, data_yaw);

		// INTEGER CONVERSION

		H_matrix = H_translate *  H_translate_distance * H_translate_altitude  *\
			   pers_proj_roll* pers_proj_pitch *  pers_proj_yaw  * H_translate_minus;

		end_matrix_calculation_time = clock();

		start_projection_time = clock();
		if (stabilization_rotation_on && stabilization_distance_on && stabilization_altitude_on) { 

			/*for (int i = 0; i < DISPLAY_RES_HEIGHT; i++){
			  for (int j = 0; j < DISPLAY_RES_WIDTH; j++){
			//for (int k = 0; k < 2; k ++){
			frame_warped.at<float>(i,j,0) = (H_matrix.at<float>(0,0)*frame.at<float>(i,j,0)+H_matrix.at<float>(0,1)*frame.at<float>(i,j,1)+H_matrix.at<float>(0,2))/(H_matrix.at<float>(2,0)*frame.at<float>(i,j,0)+H_matrix.at<float>(2,1)*frame.at<float>(i,j,1)+H_matrix.at<float>(2,2));
			frame_warped.at<float>(i,j,1) = (H_matrix.at<float>(1,0)*frame.at<float>(i,j,0)+H_matrix.at<float>(1,1)*frame.at<float>(i,j,1)+H_matrix.at<float>(1,2))/(H_matrix.at<float>(2,0)*frame.at<float>(i,j,0)+H_matrix.at<float>(2,1)*frame.at<float>(i,j,1)+H_matrix.at<float>(2,2));
			//} 
			}

			}*/


			transform_ROI(frame, frame_warped, H_matrix, frame_res, INTER_LINEAR, BORDER_CONSTANT, Scalar(), Point(0,0));

			imshow("FRAME", frame_warped); 

		} else {

			imshow("FRAME", frame); 
		}


		end_projection_time = clock();


		char ch = waitKey(15); 
		if (ch == 'q') { // exit program
			return -1;

		} else if (ch == 'i') { // re-initialize
			is_roll_initialized     = false;
			is_pitch_initialized    = false;
			is_yaw_initialized      = false;
			is_distance_initialized = false;
			is_altitude_initialized = false;

		} else if (ch == 'r') { // toggle rotational stabilization on/off
			stabilization_rotation_on = !stabilization_rotation_on;

		} else if (ch == '1') { // change image
			image_input = imread(media_list[0].filename, cv::IMREAD_COLOR);
			media_mode = media_list[0].mode;

			resize(image_input, resized_media, media_res); 
			resized_media.copyTo(frame(Rect(FRAME_ROI_LEFT, FRAME_ROI_TOP, MEDIA_RES_WIDTH, MEDIA_RES_HEIGHT)));

		} else if (ch == '2') { // change image
			image_input = imread(media_list[1].filename, cv::IMREAD_COLOR);
			media_mode = media_list[1].mode;
			resize(image_input, resized_media, media_res); 
			resized_media.copyTo(frame(Rect(FRAME_ROI_LEFT, FRAME_ROI_TOP, MEDIA_RES_WIDTH, MEDIA_RES_HEIGHT)));
		} else if (ch == '3') { // change image
			image_input = imread(media_list[2].filename, cv::IMREAD_COLOR);
			media_mode = media_list[2].mode;
			resize(image_input, resized_media, media_res); 
			resized_media.copyTo(frame(Rect(FRAME_ROI_LEFT, FRAME_ROI_TOP, MEDIA_RES_WIDTH, MEDIA_RES_HEIGHT)));

		} else if (ch == '4') { // change image
			image_input = imread(media_list[3].filename, cv::IMREAD_COLOR);
			media_mode = media_list[3].mode;

			resize(image_input, resized_media, media_res); 
			resized_media.copyTo(frame(Rect(FRAME_ROI_LEFT, FRAME_ROI_TOP, MEDIA_RES_WIDTH, MEDIA_RES_HEIGHT)));


		} else if (ch == '5') { // change video
			video_cap = VideoCapture(media_list[4].filename);

			if (!video_cap.isOpened()) {
				//printf("Video open error!!!\n");
				exit(0);
			}

			media_mode = media_list[4].mode;

			frame_cnt = 0;
		}
		else if (ch == 'd') { // toggle scaling stabilization on/off
			stabilization_distance_on = !stabilization_distance_on;
		}
		else if (ch == 'a'){  // toggle fluctutation stabilization on/off
			stabilization_altitude_on = !stabilization_altitude_on;
		}


		if (outfile_sensors.is_open ()) {
			// cout<<"\n\n outfile for sensors Opens Successfully"<<endl;
			outfile_sensors << frame_cnt << "," << Roll << "," << Pitch << "," << Yaw << "," << distance  << "," << altitude << endl;
		}

		if (outfile_matrix.is_open ()) {
			// cout<<"\n\n outfile for matrix Opens Successfully"<<endl;

			outfile_matrix << frame_cnt << endl; 
			outfile_matrix << H_matrix.at<float>(0, 0) << "," << H_matrix.at<float>(0, 1) << "," << H_matrix.at<float>(0, 2)<< endl;
			outfile_matrix << H_matrix.at<float>(1, 0) << "," << H_matrix.at<float>(1, 1) << "," << H_matrix.at<float>(1, 2)<< endl;
			outfile_matrix << H_matrix.at<float>(2, 0) << "," << H_matrix.at<float>(2, 1) << "," << H_matrix.at<float>(2, 2)<< endl << endl;

		}

		if (outfile_matrix_calculation_gain_time.is_open ()) {
			//cout<<"\n\n outfile for matrix calculatin gain time Opens Successfully"<<endl;

			outfile_matrix_calculation_gain_time << frame_cnt << ","<< ((float)(end_matrix_calculation_time-start_matrix_calculation_time) / CLOCKS_PER_SEC) << endl;


		}

		if (outfile_projection_gain_time.is_open ()) {
			//cout<<"\n\n outfile for projection gain time Opens Successfully"<<endl;
			outfile_projection_gain_time << frame_cnt << "," << ((float)(end_projection_time-start_projection_time) / CLOCKS_PER_SEC) << endl;


		}

		//  if (outfile_frame_per_sec.is_open ()){
		//    cout<<"\n\n outfile for total time Opens Successfully"<<endl;
		//    outfile_frame_per_sec << frame_cnt << "," << ((float)(end_frame_per_sec-start_frame_per_sec) / CLOCKS_PER_SEC) << endl;
		// }

		end_total_time = clock();

		if (outfile_total_time.is_open ()){
			//cout<<"\n\n outfile for total time Opens Successfully"<<endl;
			outfile_total_time << frame_cnt << "," << ((float)(end_total_time-start_total_time) / CLOCKS_PER_SEC) << endl;
		}
	}


	g_bRunThreadAHRS     = false;
	g_bRunThreadAltitude = false;
	g_bRunThreadDistance = false;


	for (i = 0 ; i < tident.size(); i++){
		pthread_join (tident[i], (void **) &status);
	}

	outfile_sensors.close ();
	outfile_matrix.close();
	outfile_matrix_calculation_gain_time.close();
	outfile_projection_gain_time.close();     
}

void delay(clock_t n)

{
	clock_t de = clock();
}

static int lookup(const char *arg)
{
	if (arg == NULL) return false;

	DIR *dirp;
	bool bExists = false;

	dirp = opendir (arg);
	if (dirp != NULL){
		bExists = true; 
	}
	return bExists;
}

char* file_create(char* filename){
	int k;


	memset (dirbuff, 0, 1024);
	memset (file_buffer, 0, 1024);
	memset (buffertmp, 0, 1024);

	getcwd(dirbuff, 1024); 

	sprintf(file_buffer,"%s/%04d-%02d-%02d_%02d:%02d:%02d",dirbuff, struct_time->tm_year+1900, struct_time->tm_mon+1, struct_time->tm_mday,    struct_time->tm_hour, struct_time->tm_min, struct_time->tm_sec);


	bool findres = lookup (file_buffer);

	if (findres == false){

		//printf("folder does not exist, so %s will be created !!!\n", file_buffer);
		if(mkdir (file_buffer, 0777) == -1 && errno != EEXIST) {
			// printf("error while trying to create %s\n", file_buffer);
			//k = -1; // abnormal exist with error
		}
		else {
			//k = 0;
			sprintf(buffertmp, "%s/%s.csv", file_buffer,filename);	
		} // normal exist with non-error
	}
	else {
		//printf("folder does already exists !!!\n");
		// k = 1; // file already exist     
		sprintf(buffertmp, "%s/%s.csv", file_buffer,filename);	   
	}	

	//delay(10);
	return buffertmp;

}

void setting_serialport(int fd, speed_t baudrate, int wait_ch)
{
	struct termios SerialPortSettings = {0};

	memset(&SerialPortSettings, 0, sizeof(SerialPortSettings));	

	// 8N1 Mode
	SerialPortSettings.c_iflag = 0;
	SerialPortSettings.c_oflag = 0;
	SerialPortSettings.c_cflag = CS8 | CREAD | CLOCAL;
	SerialPortSettings.c_lflag = 0;


	// Setting Time outs
	SerialPortSettings.c_cc[VMIN] = wait_ch; // Read at least 64 characters
	SerialPortSettings.c_cc[VTIME] = 0; // Wait indefinetly TIME * 0.1

	// Setting the Baud rate
	cfsetispeed(&SerialPortSettings, baudrate); 
	cfsetospeed(&SerialPortSettings, baudrate); 


	if((tcsetattr(fd,TCSANOW,&SerialPortSettings)) != 0) { 
		//printf("\n  ERROR ! in Setting attributes\n");
	} else {
		//printf("\n  BaudRate = %d \n  StopBits = 1 \n  Parity   = none\n", baudrate);
	}
}

void *threadforAHRS(void* argumentPointer){ // AHRS sensor

	int fdAHRS;   // File Descriptor

	//printf("\n +-----------------------------------------+");
	//printf("\n |        Serial Port Read (AHRS)          |");
	//printf("\n +-----------------------------------------+\n");

	// Opening the Serial Port

	fdAHRS = open(UART_PORT_AHRS, O_RDWR | O_NOCTTY); 


	if(fdAHRS == -1) {  // Error Checking
		//printf("Error! in Opening %s\n", UART_PORT_AHRS);
		return NULL;
	}
	else {
		//printf("%s Opened Successfully ", UART_PORT_AHRS);
	}


	setting_serialport(fdAHRS, B115200, 48);

	char    *tmp;
	char    delimiter[] = "*,'=?";
	float   split_data[6]; // Roll angle, Pitch angle, Yaw angle, X, Y, Z
	int     bytes_read = 0; // Number of bytes read by the read() system call
	int     i = 0;
	//int opt = 0; // 1: roll, 2: pitch, 3: yaw

	float Roll = 0, Pitch = 0, Yaw = 0, dist_x = 0, dist_y = 0, dist_z = 0;
	int Roll_sign = 1, Pitch_sign = 1, Yaw_sign = 1;

	float Roll_rad = 0, Pitch_rad = 0, Yaw_rad = 0;

	tcflush(fdAHRS, TCIOFLUSH); 

	char  read_buffer_ahrs[READ_BUF_SIZE_AHRS]= "";

	char* buffer2 = file_create (filename[2]);	
	fstream outfile_rotation_sensors_gain_time(buffer2, std::fstream::out | std::fstream::app);

	while(g_bRunThreadAHRS) {  
		start_rotation_sensors_time = clock();

		memset(read_buffer_ahrs,0, READ_BUF_SIZE_AHRS);
		tcflush(fdAHRS, TCIFLUSH); 


		bytes_read = read(fdAHRS, read_buffer_ahrs,READ_BUF_SIZE_AHRS);

		if(bytes_read <= 0) continue;


		i = 0;

		// data is split per (,) unit
		tmp = strtok(read_buffer_ahrs, delimiter);
		if(tmp) {
			split_data[i] = atof(tmp); 
			//printf("%f\n", split_data[i]);
			i++;    
		} 
		//	while(tmp != NULL){
		while(i < 6) {
			tmp = strtok(NULL, delimiter);
			if(tmp) {
				split_data[i] = atof(tmp); 
				//printf("%f\n", split_data[i]);
				i++;    
			} 
		}


		Roll   = split_data[0]; Pitch  = split_data[1]; Yaw    = split_data[2]; 

		if (is_roll_initialized == false) {
			initial_roll = Roll;  is_roll_initialized = true;
		}

		Roll = Roll-initial_roll;

		// Pitch error compensation
		if (is_pitch_initialized == false) {
			initial_pitch = Pitch;  is_pitch_initialized = true;
		}
		Pitch = Pitch-initial_pitch;

		// Yaw error compensation
		if (is_yaw_initialized == false) {
			initial_yaw = Yaw;  is_yaw_initialized = true;
		}

		Yaw = Yaw-initial_yaw;

		if (prev_real_roll != Roll){

			prev_real_roll = sensors[SENSORS_IDX_ROLL];
			sensors[SENSORS_IDX_ROLL] = Roll;

		}
		if (prev_real_pitch != Pitch){

			prev_real_pitch = sensors[SENSORS_IDX_PITCH];
			sensors[SENSORS_IDX_PITCH] = Pitch;

		}

		if (prev_real_yaw != Yaw){

			prev_real_yaw = sensors[SENSORS_IDX_YAW]; 
			sensors[SENSORS_IDX_YAW] = Yaw;

		}

		end_rotation_sensors_time = clock();



		if (outfile_rotation_sensors_gain_time.is_open ()) {

			// cout<<"\n\n outfile for rotation sensor time Opens Successfully"<<endl;
			outfile_rotation_sensors_gain_time <<  frame_cnt << "," << ((float)(end_rotation_sensors_time-start_rotation_sensors_time) / CLOCKS_PER_SEC) << endl;

		}

	}
	outfile_rotation_sensors_gain_time.close();

	tcflush(fdAHRS, TCIOFLUSH);
	close(fdAHRS);  


	}
	void* threadforDistance(void *argumentPointer){ 


		const unsigned char  HEADER = 0x59;

		float distance_dist_value, distance_strength_value;
		unsigned char  a, b, c ,d , e, f, g, h;
		unsigned char  checksum; 
		char  read_buffer_distance[READ_BUF_SIZE_DISTANCE] = "";
		//CirCularQueue cq; 
		//  int front = -1, rear = -1, n = 5;

		//	printf("\n +-----------------------------------------+");
		//	printf("\n |  Serial Port Read (Distance, LIDAR)     |");
		//	printf("\n +-----------------------------------------+\n");


		int fdDistance = open(UART_PORT_DISTANCE, O_RDWR | O_NOCTTY);


		//  fdDistance = open("/dev/ttyUSB1", O_RDWR | O_NOCTTY); 
		if(fdDistance == -1) {  // Error Checking
			//printf("Error! in Opening %s\n", UART_PORT_DISTANCE);
			return NULL;

		} else {
			//printf("%s Opened Successfully\n", UART_PORT_DISTANCE);
		}

		setting_serialport(fdDistance, B115200, 9);
		tcflush(fdDistance, TCIOFLUSH); 

		bool flag = false;
		cnt_for_distance = 0;
		//bool checker = false;


		int bytes_read = 0;

		char* buffer3 = file_create (filename[3]);	
		fstream outfile_distance_sensors_gain_time (buffer3, std::fstream::out | std::fstream::app);
		while (g_bRunThreadDistance) {

			start_distance_sensors_time = clock();
			//printf("distance loop in\n"); 
			memset(read_buffer_distance, 0, READ_BUF_SIZE_DISTANCE);
			tcflush(fdDistance, TCIFLUSH);  

			//flag = true;
			//printf("distance before read0\n");
			bytes_read = read(fdDistance, read_buffer_distance, READ_BUF_SIZE_DISTANCE);


			///printf("distance after read0\n");
			if(bytes_read <= 0) { /*printf("Distance: bytes_read <= 0\n");*/ continue; } // no data


			if ((read_buffer_distance[0] != HEADER) ||
					(read_buffer_distance[1] != HEADER)) {
				// printf("Distance: HEADER is not correct\n");
				continue;
			} //header is incorrect

			c = read_buffer_distance[2];
			d = read_buffer_distance[3];
			e = read_buffer_distance[4];
			f = read_buffer_distance[5];
			g = read_buffer_distance[6];
			h = read_buffer_distance[7];

			// read checksum byte
			checksum = HEADER + HEADER + c + d + e + f + g + h;

			unsigned  char tmp = checksum & 0xFF;

			if (read_buffer_distance[8] != tmp) {  // checksum is incorrect
				//  printf("Distance: Checksum is not correct buf: %x calc: %x\n", read_buffer_distance[8], tmp);
				continue;
			}

			distance_dist_value = ((int)c + (int)d * 256);
			if (prev_real_distance == distance_dist_value ) {

				continue;
			}

			// distance  error compensation
			if (is_distance_initialized == false) {
				if ( distance_dist_value > 1200 || distance_dist_value < 30) continue;
				initial_distance = distance_dist_value;  is_distance_initialized = true;
			}

			if (is_distance_initialized == true){
				if (sensors[1] > angle_threshold || sensors[2] > angle_threshold) {
					distance_dist_value = prev_real_distance;             
				}

			}

			distance_strength_value = ((int)e + (int)f * 256);
			cnt_for_distance++;

			prev_real_distance = distance_dist_value;		

			sensors[SENSORS_IDX_DISTANCE] = distance_dist_value;		

			end_distance_sensors_time = clock();


			if (outfile_distance_sensors_gain_time.is_open ()) {

				//cout<<"\n\n outfile for distance sensor time Opens Successfully"<<endl;
				outfile_distance_sensors_gain_time << frame_cnt << "," << ((float)(end_distance_sensors_time-start_distance_sensors_time) / CLOCKS_PER_SEC) << endl; 

			}

		}		
		outfile_distance_sensors_gain_time.close();
		tcflush(fdDistance, TCIOFLUSH); 
		close(fdDistance);
	}

	void* threadforAltitude(void *argumentPointer){ 

		const unsigned char  HEADER = 0x59;

		float altitude_value, altitude_strength;
		unsigned char  a, b, c ,d , e, f, g, h;
		unsigned char  checksum; 
		char  read_buffer_altitude[READ_BUF_SIZE_ALTITUDE] = "";
		//CirCularQueue cq; 
		//  int front = -1, rear = -1, n = 5;

		//	printf("\n +-----------------------------------------+");
		//	printf("\n |  Serial Port Read (Altitude, LIDAR)     |");
		//	printf("\n +-----------------------------------------+\n");


		int fdAltitude = open(UART_PORT_ALTITUDE, O_RDWR | O_NOCTTY);


		//  fdAltitude = open("/dev/ttyUSB1", O_RDWR | O_NOCTTY); 
		if(fdAltitude == -1) {  // Error Checking
			//	printf("Error! in Opening %s\n", UART_PORT_ALTITUDE);
			return NULL;

		} else {
			//	printf("%s Opened Successfully\n", UART_PORT_ALTITUDE);
		}

		setting_serialport(fdAltitude, B115200, 9);
		tcflush(fdAltitude, TCIOFLUSH); 

		bool flag = false;
		cnt_for_altitude = 0;
		//bool checker = false;


		int bytes_read = 0;


		char* buffer4 = file_create (filename[4]);	
		fstream outfile_altitude_sensors_gain_time(buffer4, std::fstream::out |  std::fstream::app);

		while (g_bRunThreadAltitude) {
			start_altitude_sensors_time = clock();


			//printf("altitude loop in\n"); 
			memset(read_buffer_altitude, 0, READ_BUF_SIZE_ALTITUDE);
			tcflush(fdAltitude, TCIFLUSH);  

			//flag = true;
			//printf("altitude before read0\n");
			bytes_read = read(fdAltitude, read_buffer_altitude, READ_BUF_SIZE_ALTITUDE);


			///printf("altitude after read0\n");
			if(bytes_read <= 0) { /*printf("Altitude: bytes_read <= 0\n");*/ continue; } // no data



			if ((read_buffer_altitude[0] != HEADER) ||
					(read_buffer_altitude[1] != HEADER)) {
				//printf("Altitude: HEADER is not correct\n");
				continue;
			} //header is incorrect

			c = read_buffer_altitude[2];
			d = read_buffer_altitude[3];
			e = read_buffer_altitude[4];
			f = read_buffer_altitude[5];
			g = read_buffer_altitude[6];
			h = read_buffer_altitude[7];

			// read checksum byte
			checksum = HEADER + HEADER + c + d + e + f + g + h;

			unsigned  char tmp = checksum & 0xFF;

			if (read_buffer_altitude[8] != tmp) {  // checksum is incorrect
				//printf("Altitude: Checksum is not correct buf: %x calc: %x\n", read_buffer_altitude[8], tmp);
				continue;
			}

			altitude_value = ((int)c + (int)d * 256);
			if (prev_real_altitude == altitude_value /*|| altitude_value > 1200 || altitude_value < 30*/) {

				continue;
			}

			// altitude  error compensation
			if (is_altitude_initialized == false) {
				if (altitude_value > 1200 || altitude_value < 30) continue;
				initial_altitude = altitude_value;  is_altitude_initialized = true;
			}
			altitude_value -= initial_altitude;

			if (is_altitude_initialized == true){
				if (sensors[1] > angle_threshold || sensors[0] > angle_threshold) {
					altitude_value = prev_real_altitude;             
				}

			}

			// calculate signal strength
			altitude_strength = ((int)e + (int)f * 256);
			cnt_for_altitude++;

			prev_real_altitude = altitude_value;		

			sensors[SENSORS_IDX_ALTITUDE] = altitude_value;		

			end_altitude_sensors_time = clock();


			if (outfile_altitude_sensors_gain_time.is_open ()) {

				//cout<<"\n\n outfile for altitude sensor time Opens Successfully"<<endl;
				outfile_altitude_sensors_gain_time << frame_cnt << "," << ((float)(end_altitude_sensors_time-start_altitude_sensors_time) / CLOCKS_PER_SEC) << endl;  


			}

		}		


		outfile_altitude_sensors_gain_time.close();
		tcflush(fdAltitude, TCIOFLUSH); 
		close(fdAltitude);
	}

	//transform_ROI(frame, frame_warped, H_matrix, frame_res, INTER_LINEAR, BORDER_CONSTANT, Scalar(), Point(0,0));
	void transform_ROI(const Mat& src, Mat& dst, const Mat& M0, Size dsize, int flags, int borderType, const Scalar& borderValue, Point origin)
	{
		dst.create( dsize, src.type() );

        dst = cv::Scalar::all(0);
		const int BLOCK_SZ = 32;
		short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];
		float M[9];
		Mat _M(3, 3, CV_32F, M);
		int interpolation = flags & INTER_MAX;

		// float ROI_left, ROI_right, ROI_top, ROI_bottom;


		if( interpolation == INTER_AREA ) interpolation = INTER_LINEAR;

		CV_Assert( (M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 && M0.cols == 3 );
		M0.convertTo(_M, _M.type());



		int x, xDest, y, yDest, x1, y1, width = dst.cols, height = dst.rows;
       
		int bh0 = std::min(BLOCK_SZ/2, height);
		int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, width);
		bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, height);

        float tmp_x, tmp_y, tmp_w0, W;
		Point Topleft(FRAME_ROI_LEFT, FRAME_ROI_TOP), Topright(FRAME_ROI_RIGHT, FRAME_ROI_TOP);
		Point Bottomleft(FRAME_ROI_LEFT, FRAME_ROI_BOTTOM), Bottomright(FRAME_ROI_RIGHT, FRAME_ROI_BOTTOM);

 
       // if( interpolation == INTER_NEAREST )
		//{
        tmp_x        = M[0]*Topleft.x + M[1]*Topleft.y + M[2];
		tmp_y        = M[3]*Topleft.x + M[4]*Topleft.y + M[5];
		tmp_w0           = M[6]*Topleft.x + M[7]*Topleft.y + M[8];
        W = tmp_w0 ? 1./tmp_w0 : 0;
        //Topleft.x        = saturate_cast<int>((Topleft.x + M[0])*W);
        //Topleft.y        = saturate_cast<int>((Topleft.y + M[0])*W);

        Topleft.x        = saturate_cast<int>((tmp_x)*W);
        Topleft.y        = saturate_cast<int>((tmp_y)*W);

        tmp_x        = M[0]*Topright.x + M[1]*Topright.y + M[2];
		tmp_y        = M[3]*Topright.x + M[4]*Topright.y + M[5];
		tmp_w0            = M[6]*Topright.x + M[7]*Topright.y + M[8];
        W = tmp_w0 ? 1./tmp_w0 : 0;
        Topright.x        = saturate_cast<int>((tmp_x)*W);
        Topright.y        = saturate_cast<int>((tmp_y)*W);
        
        tmp_x        = M[0]*Bottomleft.x + M[1]*Bottomleft.y + M[2];
		tmp_y        = M[3]*Bottomleft.x + M[4]*Bottomleft.y + M[5];
		tmp_w0              = M[6]*Bottomleft.x + M[7]*Bottomleft.y + M[8];
        W = tmp_w0 ? 1./tmp_w0 : 0;
        Bottomleft.x        = saturate_cast<int>((tmp_x)*W);
        Bottomleft.y        = saturate_cast<int>((tmp_y)*W);
        
        tmp_x        = M[0]*Bottomright.x + M[1]*Bottomright.y + M[2];
		tmp_y       = M[3]*Bottomright.x + M[4]*Bottomright.y + M[5];
		tmp_w0               = M[6]*Bottomright.x + M[7]*Bottomright.y + M[8];
        W = tmp_w0 ? 1./tmp_w0 : 0;
        Bottomright.x        = saturate_cast<int>((tmp_x)*W);
        Bottomright.y        = saturate_cast<int>((tmp_y)*W); 

        /*}						
		else
		{
        //short* alpha = A ;
        Topleft.x        = M[0]*Topleft.x + M[1]*Topleft.y + M[2];
		Topleft.y        = M[3]*Topleft.x + M[4]*Topleft.y + M[5];
		tmp_w0           = M[6]*Topleft.x + M[7]*Topleft.y + M[8];
        W = tmp_w0 ? INTER_TAB_SIZE/W : 0;
        Topleft.x        = saturate_cast<int>((Topleft.x + M[0])*W);
        Topleft.y        = saturate_cast<int>((Topleft.y + M[0])*W);
        Topleft.x        = (Topleft.x >> INTER_BITS) + origin.x;
        Topleft.y        = (Topleft.y >> INTER_BITS) + origin.y;
       // alpha[x1]        = (short)((Topleft.y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
									//(Topleft.x & (INTER_TAB_SIZE-1)));

        Topright.x        = M[0]*Topright.x + M[1]*Topright.y + M[2];
		Topright.y        = M[3]*Topright.x + M[4]*Topright.y + M[5];
		tmp_w0            = M[6]*Topright.x + M[7]*Topright.y + M[8];
        W = tmp_w0 ? INTER_TAB_SIZE/W : 0;
        Topright.x        = saturate_cast<int>((Topright.x + M[0])*W);
        Topright.y        = saturate_cast<int>((Topright.y + M[0])*W);
        Topright.x        = (Topright.x >> INTER_BITS) + origin.x;
        Topright.y        = (Topright.y >> INTER_BITS) + origin.y;
       // alpha[x1]        = (short)((Topleft.y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
								//	(Topleft.x & (INTER_TAB_SIZE-1)));
        
        Bottomleft.x        = M[0]*Bottomleft.x + M[1]*Bottomleft.y + M[2];
		Bottomleft.y        = M[3]*Bottomleft.x + M[4]*Bottomleft.y + M[5];
		tmp_w0              = M[6]*Bottomleft.x + M[7]*Bottomleft.y + M[8];
        W = tmp_w0 ? INTER_TAB_SIZE/W : 0;
        Bottomleft.x        = saturate_cast<int>((Bottomleft.x + M[0])*W);
        Bottomleft.y        = saturate_cast<int>((Bottomleft.y + M[0])*W);
        Bottomleft.x        = (Bottomleft.x >> INTER_BITS) + origin.x;
        Bottomleft.y        = (Bottomleft.y >> INTER_BITS) + origin.y;
        //alpha[x1]        = (short)((Topleft.y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
								//	(Topleft.x & (INTER_TAB_SIZE-1)));
        
        Bottomright.x        = M[0]*Bottomright.x + M[1]*Bottomright.y + M[2];
		Bottomright.y        = M[3]*Bottomright.x + M[4]*Bottomright.y + M[5];
		tmp_w0               = M[6]*Bottomright.x + M[7]*Bottomright.y + M[8];
        W = tmp_w0 ? INTER_TAB_SIZE/W : 0;
        Bottomright.x        = saturate_cast<int>((Bottomright.x + M[0])*W);
        Bottomright.y        = saturate_cast<int>((Bottomright.y + M[0])*W);
        Bottomright.x        = (Bottomright.x >> INTER_BITS) + origin.x;
        Bottomright.y        = (Bottomright.y >> INTER_BITS) + origin.y; 
        //alpha[x1]            = (short)((Topleft.y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
									//(Topleft.x & (INTER_TAB_SIZE-1)));
						

		}*/

        int max_top        = max(0, min(height, Topleft.y < Topright.y ? Topleft.y : Topright.y));
        int max_bottom     = max(0, min(height, Bottomleft.y > Bottomright.y ? Bottomleft.y : Bottomright.y));
        int max_left       = max(0, min(width,  Topleft.x < Bottomleft.x ? Topleft.x : Bottomleft.x));
        int max_right      = max(0, min(width,  Topright.x > Bottomright.x ? Topright.x : Bottomright.x));

        //printf ("\n\n111: max_top : %d, max_bottom : %d, max_left : %d, max_right : %d\n\n", max_top, max_bottom, max_left, max_right);

        max_top = min(FRAME_ROI_TOP, max_top);
        max_bottom = max(FRAME_ROI_BOTTOM, max_bottom);
        max_left = min(FRAME_ROI_LEFT, max_left);
        max_right = max(FRAME_ROI_RIGHT, max_right);
        //max_top -= 1;
       // max_bottom -= 1;
        //max_left -= 1;
        //max_right -= 2;

      //  printf ("\n\n222: max_top : %d, max_bottom : %d, max_left : %d, max_right : %d\n\n", max_top, max_bottom, max_left, max_right);
		cv::rectangle(dst, Point(max_left-1, max_top-1), Point(max_right+2, max_bottom+2), cv::Scalar(0, 255, 0));

		if( !(flags & WARP_INVERSE_MAP) ) invert(_M, _M);

		for( y = max_top , yDest = max_top; y < max_bottom; y += bh0, yDest += bh0 )
		{
			for( x = max_left, xDest = max_left; x < max_right; x += bw0, xDest += bw0 )
			{
				int bw = std::min( bw0, width  - x );
				int bh = std::min( bh0, height - y );
				// to avoid dimensions errors
				if (bw <= 0 || bh <= 0) break;

				Mat _XY(bh, bw, CV_16SC2, XY), _A;
				Mat dpart(dst, Rect(xDest, yDest, bw, bh));

				for( y1 = 0; y1 < bh ; y1++ )
				{
					short* xy = XY + y1*bw*2;
					float  X0 = M[0]*x + M[1]*(y + y1) + M[2];
					float  Y0 = M[3]*x + M[4]*(y + y1) + M[5];
					float  W0 = M[6]*x + M[7]*(y + y1) + M[8];

					if( interpolation == INTER_NEAREST )
					{
						for( x1 = 0; x1 < bw ; x1++ )						
						{
							float W = W0 + M[6]*x1;
							W = W ? 1./W : 0;
							int X = saturate_cast<int>((X0 + M[0]*x1)*W);
							int Y = saturate_cast<int>((Y0 + M[3]*x1)*W);
							xy[x1*2]   = (short)X;
							xy[x1*2+1] = (short)Y;

						}
					}
					else
					{
						short* alpha = A + y1*bw;
						for( x1 = 0; x1 < bw ; x1++ )
						{
							float W = W0 + M[6]*x1;
							W = W ? INTER_TAB_SIZE/W : 0;
							int X = saturate_cast<int>((X0 + M[0]*x1)*W);
							int Y = saturate_cast<int>((Y0 + M[3]*x1)*W);
							xy[x1*2]   = (short)(X >> INTER_BITS) + origin.x;
							xy[x1*2+1] = (short)(Y >> INTER_BITS) + origin.y;
							alpha[x1]  = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
									(X & (INTER_TAB_SIZE-1)));


						}
					}


				}

				if( interpolation == INTER_NEAREST ) {
					remap( src, dpart, _XY, Mat(), interpolation, borderType, borderValue );
				}
				else
				{
					Mat _A(bh, bw, CV_16U, A);
					remap( src, dpart, _XY, _A,    interpolation, borderType, borderValue );

				}
			}

	}
	
}

void transform_ori(const Mat& src, Mat& dst, const Mat& M0, Size dsize, int flags, int borderType, const Scalar& borderValue, Point origin )
	{
		dst.create( dsize, src.type() );

		const int BLOCK_SZ = 32;
		short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];
		float M[9];
		Mat _M(3, 3, CV_32F, M);
		int interpolation = flags & INTER_MAX;
	
		if( interpolation == INTER_AREA ) interpolation = INTER_LINEAR;

		CV_Assert( (M0.type() == CV_32F || M0.type() == CV_64F) && M0.rows == 3 && M0.cols == 3 );
		M0.convertTo(_M, _M.type());

	    if( !(flags & WARP_INVERSE_MAP) ) invert(_M, _M);

		int x, xDest, y, yDest, x1, y1, width = dst.cols, height = dst.rows;

		int bh0 = std::min(BLOCK_SZ/2, height);
		int bw0 = std::min(BLOCK_SZ*BLOCK_SZ/bh0, width);
		bh0 = std::min(BLOCK_SZ*BLOCK_SZ/bw0, height);

		for( y = -origin.y, yDest = 0; y < height; y += bh0, yDest += bh0 )
		{
			for( x = -origin.x, xDest = 0; x < width; x += bw0, xDest += bw0 )
			{
				int bw = std::min( bw0, width  - x );
				int bh = std::min( bh0, height - y );
				// to avoid dimensions errors
				if (bw <= 0 || bh <= 0)
					break;

				Mat _XY(bh, bw, CV_16SC2, XY), _A;
				Mat dpart(dst, Rect(xDest, yDest, bw, bh));

				for( y1 = 0; y1 < bh; y1++ )
				{
					short* xy = XY + y1*bw*2;
					float X0 = M[0]*x + M[1]*(y + y1) + M[2];
					float Y0 = M[3]*x + M[4]*(y + y1) + M[5];
					float W0 = M[6]*x + M[7]*(y + y1) + M[8];

					if( interpolation == INTER_NEAREST )
						for( x1 = 0; x1 < bw; x1++ )
						{
							float W = W0 + M[6]*x1;
							W = W ? 1./W : 0;
							int X = saturate_cast<int>((X0 + M[0]*x1)*W);
							int Y = saturate_cast<int>((Y0 + M[3]*x1)*W);
							xy[x1*2] = (short)X;
							xy[x1*2+1] = (short)Y;
						}
					else
					{
						short* alpha = A + y1*bw;
						for( x1 = 0; x1 < bw; x1++ )
						{
							float W = W0 + M[6]*x1;
							W = W ? INTER_TAB_SIZE/W : 0;
							int X = saturate_cast<int>((X0 + M[0]*x1)*W);
							int Y = saturate_cast<int>((Y0 + M[3]*x1)*W);
							xy[x1*2] = (short)(X >> INTER_BITS) + origin.x;
							xy[x1*2+1] = (short)(Y >> INTER_BITS) + origin.y;
							alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE +
									(X & (INTER_TAB_SIZE-1)));
						}
					}
				}

				if( interpolation == INTER_NEAREST )
					remap( src, dpart, _XY, Mat(), interpolation, borderType, borderValue );
				else
				{
					Mat _A(bh, bw, CV_16U, A);
					remap( src, dpart, _XY, _A, interpolation, borderType, borderValue );
				}
			}
		}
	}

