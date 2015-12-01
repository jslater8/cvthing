/////////////////////////////////////////////////////////////////////////////
//
// COMS30121 - blur.cpp
// TOPIC: basic image blur via convolution with Gaussian Kernel
//
/////////////////////////////////////////////////////////////////////////////

// header inclusion
#include <stdio.h>
#include <math.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace cv;
using namespace std;

#define nrows 3
#define ncols 3


void sobel(Mat &input, Mat &dxOutput, Mat &dyOutput, Mat &magnitude, Mat &angle);

Mat convo(
		Mat &input,
		int kernel[nrows][ncols]);

Mat hough(Mat &input);
Mat houghline(Mat &input);
Mat houghcircle(Mat &input);

void mag(cv::Mat &dx, cv::Mat &dy, cv::Mat &output);
void getAngle(cv::Mat &dx, cv::Mat &dy, cv::Mat &output);
int detect(Mat &in, Mat &line, Mat &circle);

int hc[1000][1000][500];
int h[1000][360];
int linePoints[100000][2][2];
int lineAngles[100000];
int angles[1000][1000];

Mat GaussianBlur(Mat &input) {
	int kernel[][3] = {{1,1,1},{1,1,1},{1,1,1}};
	Mat out;
	cvtColor( input, out, CV_BGR2GRAY );
	return convo(out, kernel);
}

int main2( int argc, char** argv )
{

 // LOADING THE IMAGE
 char* imageName = argv[1];

 Mat image;
 image = imread( imageName, 1 );

 if( argc != 2 || !image.data )
 {
   printf( " No image data \n " );
   return -1;
 }


	//Mat houghspace = houghline(image);

	// Mat blur = GaussianBlur(image);
	Mat blur;
	cv::GaussianBlur(image,blur,Size(11,11),0);
	printf("test1");
	fflush(stdout);
	Mat hcircle = houghcircle(blur);
	printf("test2");
	fflush(stdout);
	Mat hline = houghline(blur);
	printf("test3");
	fflush(stdout);
	detect(image, hline, hcircle);


 	return 0;
}

int doDetect(Mat &im) {
	Mat image = im.clone();
	Mat blur = GaussianBlur(image);
	// printf("test1");
	// fflush(stdout);
	Mat hcircle = houghcircle(blur);
	// printf("test2");
	// fflush(stdout);
	Mat hline = houghline(blur);
	// printf("test3");
	// fflush(stdout);
	return detect(image, hline, hcircle);
}



void sobel(Mat &input, Mat &dxOutput, Mat &dyOutput, Mat &magnitude, Mat &angle){
	int dx[][ncols] = {{-1,0,1},{-2,0,2},{-1,0,1}};
	dxOutput = convo(input, dx);

	int dy[][ncols] = {{-1,-2,-1},{0,0,0},{1,2,1}};
	dyOutput = convo(input, dy);

	mag(dxOutput, dyOutput, magnitude);

	getAngle(dxOutput,dyOutput,angle);
}

void getAngle(cv::Mat &dx, cv::Mat &dy, cv::Mat &output){
	output.create(dx.size(), dx.type());
	for (int i = 0; i < dx.rows; i++){
		for (int j = 0; j < dx.cols; j++){
			float value = 0;
			output.at<uchar>(i,j) = 0;


			float value1 = float(dy.at<uchar>(i,j) -128.0);
			float value2 = float(dx.at<uchar>(i,j) -128.0);
			output.at<uchar>(i,j) = (uchar)255*(((float)(atan2(value1,value2)*180/3.1416) + 180)/360.0);
			// printf("%f\n", (atan2(value1, value2)*180.0/3.1416) + 180.0);
			//if (value1 == 0.0 && value2 == 0.0) printf("%d, %d\n",i,j);





			/*if (dy.at<uchar>(i,j) > 128 && dx.at<uchar>(i,j) < 128) {
				printf("%d %d\n", dy.at<uchar>(i,j), dx.at<uchar>(i,j));
				value = float(dy.at<uchar>(i,j))/(float(dx.at<uchar>(i,j)));
				output.at<uchar>(i,j) = (uchar)((atan(value)*180/3.1415));
			} else if (dy.at<uchar>(i,j) > 128 && dx.at<uchar>(i,j) > 128) {
				value = float(dy.at<uchar>(i,j))/(float(dx.at<uchar>(i,j)));
				output.at<uchar>(i,j) = (uchar)((atan(value)*180/3.1415));
			//	printf("test");
			} else if (dy.at<uchar>(i,j) < 128 && dx.at<uchar>(i,j) > 128) {
				value = float(128+dy.at<uchar>(i,j))/(float(dx.at<uchar>(i,j)));
				output.at<uchar>(i,j) = (uchar)((atan(value)*180/3.1415));
			} else {
				value = float(dy.at<uchar>(i,j))/float(dx.at<uchar>(i,j));
				output.at<uchar>(i,j) = (uchar)((atan(value)*180/3.1415));
			//	printf("test2");*/
		//}

		}

	}
}

void mag(cv::Mat &dx, cv::Mat &dy, cv::Mat &output){
	output.create(dx.size(), dx.type());
	for (int i = 0; i < dx.rows; i++){
		for (int j = 0; j < dx.cols; j++){
			int value = dx.at<uchar>(i,j) * dx.at<uchar>(i,j) + dy.at<uchar>(i,j) * dy.at<uchar>(i,j);
			output.at<uchar>(i,j) = (uchar)sqrt(float(value));
			//printf("%d\n", (uchar)sqrt(float(value)));
		}
	}
}

int detect(Mat &in, Mat &line, Mat &circle) {


	int maxc = 0;
	int maxl = 0;
	int totmax = 0;
	Point circlepos;
	int count = 0;
	int total = 0;
	int totalmax = 0;
	int size = in.cols * in.rows;
	for (int i = 0; i < in.rows-5; i++) {
		for (int j = 0; j < in.cols-5; j++) {
		//	in.at<Vec3b>(i,j)[2] = 255;
			int lcount = 0;
			int ccount = 0;
			int lmax = 0;
			int cmax = 0;
			for (int x = 0; x<5; x++) {
				for (int y = 0; y<5; y++) {
					lcount += line.at<uchar>(i+x,j+y);
					if (line.at<uchar>(i+x,j+y) > lmax)
						lmax = line.at<uchar>(i+x,j+y);
					ccount += circle.at<uchar>(i+x,j+y);
					if (circle.at<uchar>(i+x,j+y) > cmax)
						cmax = circle.at<uchar>(i+x,j+y);
				}
			}
			if (lmax*500 > size && cmax*400 > size) {
				// printf("%d %d\n", lcount, ccount);
				return 1;
				count++;

				circlepos.x = j;
				circlepos.y = i;
				// for (int x = 0; x<in.rows/20; x++) {
				// 	for (int y = 0; y<in.cols/5; y++) {
				// 		 in.at<Vec3b>(i+x,j+y)[0] = 255;
				// 	}
				// }
			}

		}
	}

	// int size = in.cols * in.rows;
	// int numlines = 0;
	// int numcircles = 0;
	// for (int i = 0; i < in.rows; i++) {
	// 	for (int j = 0; j < in.cols; j++) {
	// 		if (line.at<Vec3b>(i,j)[0]){
	// 			numlines++;
	// 		}
	// 		if (circle.at<uchar>(i,j)){
	// 			numcircles++;
	// 		}
	// 	}
	// }
	// std::cout << "numlines "  << numlines << ", numcricles " << numcircles << ", size " << size << std::endl;

	// if (numcircles*3 > size && numlines*500 > size) return 1;
	return 0;

	imwrite("output.jpg", in);
	imwrite("circle.jpg", circle);
	imwrite("line.jpg", line);
}

Mat houghline(Mat &input) {



	int dx[][ncols] = {{-1,0,1},{-2,0,2},{-1,0,1}};
	Mat dxOutput = convo(input, dx);

	imwrite("dx.jpg", dxOutput);

	int dy[][ncols] = {{-1,-2,-1},{0,0,0},{1,2,1}};
	Mat dyOutput = convo(input, dy);
	imwrite("dy.jpg", dyOutput);

	Mat magnitude;
	Canny( input, magnitude, 50, 185, 3 );

	imwrite("mag.jpg",magnitude);



	Mat angle;
	getAngle(dxOutput,dyOutput,angle);



	imwrite("angle.jpg", angle);

	int maxro = sqrt(pow(input.rows,2)+ pow(input.cols,2));



	for (int a = 0; a < 360; a++) {
		for (int b = 0; b < maxro; b++ ) {
			h[b][a] = 0;
		}
	}



int count = 0;
	for (int i = 0; i < magnitude.rows; i++) {
		for (int j = 0; j < magnitude.cols; j++) {
			if (magnitude.at<uchar>(i,j) >= 185) {
				int mina = 360;
				int maxa = 0;
				for (int a = 0; a < 360; a++) {

					//	if (angle.at<uchar>(i,j) >= a - 20 && angle.at<uchar>(i,j) <= a + 20) {

						int b = (int)(i*sin(((a)-180.0)*3.1416/180.0) + j*cos(((a)-180.0)*3.1416/180.0));
						if (b < maxro) {
							h[b][a] += 1;

						}

				//	}

				}

			}
		}
	}

	Mat outline(maxro, 360, CV_8UC1);
   for (int i = 0; i < maxro; i++) {
	   for (int j = 0; j < 360; j++) {
		   outline.at<uchar>(i,j) = 0;
	   }
   }



	float lines[100000][2];
	int count2 = 0;

	for (int i = 1; i < maxro; i++) {
		for (int j = 0; j < 360; j++) {
			if (h[i][j] > 50 && count2 < 100000) {
				outline.at<uchar>(i,j) = h[i][j];
				lines[count2][0] = i;
				lines[count2][1] = j;
				count2++;
			}
		}
	}
	 imwrite("outline.jpg", outline);


	for (int i = 0; i < count2; i++) {
		float rho = lines[i][0], theta = lines[i][1];
	 Point pt1, pt2;
	 double a = cos(((theta)-180)*3.1416/180.0), b = sin(((theta)-180)*3.1416/180.0);
	 double x0 = a*rho, y0 = b*rho;
	 pt1.x = cvRound(x0 + 1000*(-b));
	 pt1.y = cvRound(y0 + 1000*(a));
	 pt2.x = cvRound(x0 - 1000*(-b));
	 pt2.y = cvRound(y0 - 1000*(a));

	 linePoints[i][0][0] = pt1.y;
	 linePoints[i][0][1] = pt1.x;
	 linePoints[i][1][0] = pt2.y;
	 linePoints[i][1][1] = pt2.x;
	 lineAngles[i] = theta;

	// line( in, pt1, pt2, Scalar(0,0,255), 1);
//	if (x0 > 0 && x0 < in.rows && y0 > 0 && y0 < in.cols)
//		in.at<Vec3b>(x0,y0)[2] = 255;

	}

	Mat output(input.rows, input.cols, CV_8UC1);

	int out[1000][1000];
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			angles[i][j] = 0;
			out[i][j] = 0;
		}
	}
	int first = 1;
	int max = 0;
	for (int i = 0; i < count2; i++) {
		int x1 = linePoints[i][0][0];
		int y1 = linePoints[i][0][1];
		int x2 = linePoints[i][1][0];
		int y2 = linePoints[i][1][1];

		for (int j = 0; j < count2; j++) {

			if (i != j) {
				int x3 = linePoints[j][0][0];
				int y3 = linePoints[j][0][1];
				int x4 = linePoints[j][1][0];
				int y4 = linePoints[j][1][1];

				int x =0;
				int y = 0;
				double test = ((x1-x2)*(y3-y4)-(y1-y2)*(x3-x4));

				if (abs(test) > 1e-5){

					x = ((x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4))/test;
					y = ((x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4))/test;
				//	printf("%d,%d\n",x,y);
					if (x > 0 && x < input.rows && y > 0 && y < input.cols) {
						out[x][y] += 1;
						if (abs(lineAngles[i] - lineAngles[j]) > angles[x][y])
							angles[x][y] = abs(lineAngles[i] - lineAngles[j]);

						if (out[x][y] > max) {
							max = out[x][y];
						}
					}
				}
			}
		}
	}

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			//printf("%d\n", angles[i][j]);
			if (max) output.at<uchar>(i,j) = ((float)angles[i][j]/360.0)+out[i][j];
			else output.at<uchar>(i,j) =0;
		}
	}

	imwrite("out.jpg", output);
	return output;
}
int count = 0;
Mat houghcircle(Mat &input) {



	int dx[][ncols] = {{-1,0,1},{-2,0,2},{-1,0,1}};
	Mat dxOutput = convo(input, dx);

	imwrite("dx.jpg", dxOutput);

	int dy[][ncols] = {{-1,-2,-1},{0,0,0},{1,2,1}};
	Mat dyOutput = convo(input, dy);
	imwrite("dy.jpg", dyOutput);



	Mat magnitude;
	//mag(dxOutput, dyOutput, magnitude);

	Canny( input, magnitude, 100, 185, 3 );
	imwrite("mag.jpg",magnitude);

	Mat angle;
	getAngle(dxOutput,dyOutput,angle);


	for (int a = 0; a < input.rows; a++) {
		for (int b = 0; b < input.cols; b++ ) {
			for (int c = 0; c < 500; c++ ) {
				hc[a][b][c] = 0;
			}
		}
	}

	Mat thresh(input.rows, input.cols, CV_8UC1);

	int count = 0;
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			thresh.at<uchar>(i,j) = 0;
			if (magnitude.at<uchar>(i,j) >= 185) {
				thresh.at<uchar>(i,j) = 255;
				for (int r = 2; r < 300; r++ ) {
					int x1 = i + r*sin(360*(((float)angle.at<uchar>(i, j))/255.0 )*3.1416/180.0);
					int x2 = i - r*sin(360*(((float)angle.at<uchar>(i, j))/255.0 )*3.1416/180.0);

					int y1 = j + r*cos(360*(((float)angle.at<uchar>(i, j))/255.0 )*3.1416/180.0);
					int y2 = j - r*cos(360*(((float)angle.at<uchar>(i, j))/255.0 )*3.1416/180.0);

					if (x1 < input.rows && y1 < input.cols && x1 > 0 && y1 > 0) {
						hc[x1][y1][r] += 1;
					}

					if (x2 < input.rows && y2 < input.cols && x2 > 0 && y2 > 0) {
						hc[x2][y2][r] += 1;
					}
				}
			}
		}
	}


	imwrite("thresh.jpg", thresh);
	int out[1000][1000];
	int max = 0;
	Mat output(input.rows, input.cols, CV_8UC1);
	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			out[i][j] = 0;
			for (int r = 2; r < 300; r++) {
				if (hc[i][j][r] > 1) {
		//			printf("circle radius %d at %d %d\n", r, i, j);
					out[i][j] += 1*hc[i][j][r];
					if (out[i][j] > max)
						max = out[i][j];
				}

		//		output.at<uchar>(i,j) += hc[i][j][r];
		//		printf("%d\n", output.at<uchar>(i,j));
			}
		}
	}

	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (max) output.at<uchar>(i,j) = 255*out[i][j]/max;
			else output.at<uchar>(i,j) =0;
		}
	}

/*	for (int i = 0; i < input.rows; i++) {
		for (int j = 0; j < input.cols; j++) {
			if (output.at<uchar>(i,j) > 80) {
				output.at<uchar>(i,j) = 255;
			} else {
				output.at<uchar>(i,j) = 0;
			}
		}
	}*/
	// char buffer[20];
	// sprintf(buffer, "circle%d.jpg", output.rows);
	// imwrite(buffer, output);
	count++;
	return output;
}

Mat hough(Mat &input) {

	Mat gray_image;
	cvtColor( input, gray_image, CV_BGR2GRAY );

	vector<Vec3f> circles;
	HoughCircles(gray_image, circles, CV_HOUGH_GRADIENT, 1, 90, 185, 60);

	Mat output;
	output.create(input.size(), CV_8UC1);

	for (int i = 0; i < output.rows; i++) {
		for (int j = 0; j < output.cols; j++) {
			output.at<uchar>(i, j) = (uchar)0;
		}
	}

	for (int i = 0; i < circles.size(); i++) {
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( input, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( input, center, radius, Scalar(0,0,255), 3, 8, 0 );

	 // output.at<uchar>(circles[i][0],circles[i][1]) = 0;
	  output.at<uchar>(circles[i][0],circles[i][1]) += (uchar)circles[i][2];
	}

	imwrite("dxoutput.jpg", input);
	imwrite("output.jpg", output);

}

Mat convo(cv::Mat &input, int kernel[nrows][ncols]){

	int k = nrows/2;
	Mat output;
	output.create(input.size(),input.type());

	Mat paddedInput;
	copyMakeBorder( input, paddedInput,
		1, 1, 1, 1,
		cv::BORDER_REPLICATE );

	int normal = 0;
	for ( int i = 0; i < k*2 + 1; i++ ) {
		for ( int j = 0; j < k*2 + 1; j++ ) {
			normal += sqrt(kernel[i][j]*kernel[i][j]);
		}
	}
	for ( int i = 0; i < input.rows; i++ ) {
		for( int j = 0; j < input.cols; j++ )  {
			float sum = 0.0;
			for( int m = -k; m <= k; m++ ) {
				for( int n = -k; n <= k; n++ ) {
					// get the values from the padded image and the kernel
					sum += (int)paddedInput.at<uchar>( i - m, j - n ) * (int)kernel[m+k][n+k];
				}
			}
			output.at<uchar>(i, j) = (uchar)((sum / (float)normal) + 129);
		}
	}
	return output;
}
