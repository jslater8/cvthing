/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - face.cpp
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "p1.cpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame, const char* outname );

/** Global variables */
String cascade_name = "dartcascade/cascade.xml";
CascadeClassifier cascade;



/** @function main */
int main( int argc, const char** argv )
{
       // 1. Read Input Image
	Mat frame = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	// 2. Load the Strong Classifier in a structure called `Cascade'
	if( !cascade.load( cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

	const char* output = argv[2];
	// 3. Detect Faces and Display Result
	detectAndDisplay( frame,output );

	// 4. Save Result Image
	const char* outname = argv[3];
	imwrite( outname, frame );
	return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame, const char* outname )
{
	// Mat image = imread("test9.jpg", 1);
	// printf("TEST: %d\n", doDetect(image));
	std::vector<Rect> faces;
	Mat frame_gray;

	// 1. Prepare Image by turning it into Grayscale and normalising lighting
	cvtColor( frame, frame_gray, CV_BGR2GRAY );
	equalizeHist( frame_gray, frame_gray );

	// 2. Perform Viola-Jones Object Detection
	cascade.detectMultiScale( frame_gray, faces, 1.6, 1, 0|CV_HAAR_SCALE_IMAGE, Size(80, 80), Size(500,500) );

       // 3. Print number of Faces found
	// std::cout << faces.size() << std::endl;

       // 4. Draw box around faces found
	
	// doDetect(frame, points);
	

	// for (int i = 0; i < faces.size(); i++){
	// 	for (int j = 0; j < points.size(); j++){
	// 		// make sure points from hough are inside viola jones
	// 		// Rect intersect = faces[i] & points[j];
	// 		// if ((intersect == points[j])){
	// 		// 	rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 255, 0, 0 ), 3);
	// 		// }
	// 		if(faces[i].contains(points[j])){
	// 			rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 0, 255 ), 4);
	// 		}
	// 	}
	// }

	Mat output = frame.clone();

	for(int i = 0; i < faces.size(); i++){
		Mat image;
		Rect image2 = faces[i] + cv::Size(faces[i].width*0.5, faces[i].height*0.5);
		if (image2.x + image2.width < frame.cols && image2.y + image2.height < frame.rows){
			image = frame(image2);
		}
		else image = frame(faces[i]);

		char buffer[20];
		sprintf(buffer, "test%d.jpg", i);
		imwrite(buffer, image);
		int test = doDetect(image);
		if (test) rectangle(output, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 0, 255 ), 4);
	}


	imwrite(outname,output);

	for( int i = 0; i < faces.size(); i++ )
	{
		rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), Scalar( 0, 255, 0 ), 2);
	}
}
