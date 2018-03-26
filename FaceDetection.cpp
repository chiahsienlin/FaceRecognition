/*----------------------------------------------------------------------------------------------------------------------------------------------* 
*	NYU-CS-6643 2017 Fall Computer Vision													*
*	NAME: Chia-Hsien Lin (chl566@nyu.edu)													*
*	ID: N17505647																*
* 	Project description. Design a face recognition system using the eigenface method you have learned in class. You will be given a set	*
*	of M training images and another set of test images. Use the training images to produce a set of eigenfaces. Then recognize the face	* 
*	in the input image using the eigenface method. Use Euclidean distance as distance measure for computing di for i=0 to M You can		* 
*	manually choose the thresholds T0 and T1 that produce the best results.									*   
*	Python, C/C++, Matlab or Java are the recommended programming languages. You can use built-in library functions to read/write/display	*
*	images, to compute eigenvalues and eigenvectors, and to perform other matrix/vector arithmetic operations, but you cannot use library	* 
*	functions to perform other steps you are required to implement in the project.   							*	
*																	 	*
*	threshold T0 =																*
*	threshold T1 = 			 													*
---------------------------------------------------------------------------------------------------------------------------------------------*/ 
#include <iostream>
#include "eigen_dir/Eigen/Dense"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <set>
#include <unordered_map>
#include <string>
#include <cmath>
using namespace Eigen;
using namespace cv;
using namespace std;
#define HEIGHT 231
#define WIDTH 195
#define T1  3.5e+16

void ShowEigenFace(string title, MatrixXd m);
void Insert_Col_MatrixA(MatrixXd &vec, MatrixXd &A, int col);
void Normalization(MatrixXd &m);
void SetTrainingImg(set<string> &TrainImg);
void SetInputImg(unordered_map<int, string> &InputImg);
double Distance(MatrixXd a, MatrixXd b);
MatrixXd Mat_to_Matrix(Mat img);

int main()
{ 
  set<string> TrainImg; 
  unordered_map<int,string> InputImg;
  
  SetTrainingImg(TrainImg);
  SetInputImg(InputImg);

  int M = TrainImg.size();
  MatrixXd A = MatrixXd::Zero(HEIGHT*WIDTH, M); 
  MatrixXd avgFace = MatrixXd::Zero(HEIGHT*WIDTH,1); 
  MatrixXd EigenValues = MatrixXd::Zero(M,M);
  MatrixXd FaceSpace = MatrixXd::Zero(HEIGHT*WIDTH, M);
  unordered_map<int, MatrixXd> LinearCombines;
  
  //Compute the avgFace
  for(auto img: TrainImg){
  	Mat src = imread(img, IMREAD_GRAYSCALE);
	imshow(img.substr(10), src);
        int row = 0;
	for(int i = 0; i < src.rows; i++){
		for(int j = 0; j < src.cols; j++){
			avgFace(row,0) += src.at<uchar>(i,j);
			row++;
		}
	}
  }
  avgFace /= (1.0*TrainImg.size());
  ShowEigenFace("avgFace", avgFace);  
  
  //Make the matrix A which is dimension of N^2 * M by subtracting the avgFace
  int col = 0;
  cout << "Eigen Space:" << endl;
  for(auto img: TrainImg){
  	Mat src = imread(img, IMREAD_GRAYSCALE);
	MatrixXd srcvec = MatrixXd::Zero(src.rows*src.cols,1); 
	srcvec = Mat_to_Matrix(src);
	srcvec -= avgFace;
	cout << "Eigen Face " << col+1 << ": " << img.substr(10) << endl;
  	Insert_Col_MatrixA(srcvec, A, col);
        col++;
  }
  cout << endl;

  //Find eigenvalues of V = A^T * A is of dimension M * M
  EigenValues = A.transpose() * A;
  //FaceSpace U = A * V
  FaceSpace = A * EigenValues;

  //Show all the eigen faces in the FaceSpace
  for(int n = 0; n < FaceSpace.cols(); n++){
  	MatrixXd e = MatrixXd::Zero(FaceSpace.rows(),1);
	for(int m = 0; m < FaceSpace.rows(); m++){
		e(m,0) = FaceSpace(m,n);
	}
	ShowEigenFace("EigenFace"+to_string(n+1), e);
  }
  
  //Training images projected onto FaceSpace
  for(int n = 0; n < A.cols(); n++){
  	MatrixXd R = MatrixXd::Zero(A.rows(),1);
	for(int m = 0; m < A.rows(); m++){
		 R(m,0) = A(m,n);
	}
	LinearCombines[n] = FaceSpace.transpose() * R;
  }
 
  //Recognition
  cout << "Result: " << endl; 
  for(int key = 0; key < InputImg.size(); key++){

  //Read Input image and convert to be eigenvector
  	Mat input = imread(InputImg[key],IMREAD_GRAYSCALE); 
	MatrixXd I = MatrixXd::Zero(input.rows*input.cols,1);
  	I = Mat_to_Matrix(input);
  //Substract avgFace from input face I
  	I -= avgFace;
	ShowEigenFace("Sub_avgFace_"+ InputImg[key].substr(10),I);
  //Project the input image onto face space
  	MatrixXd OmegaI = MatrixXd::Zero(M,1);
  	OmegaI = FaceSpace.transpose() * I;
//  	cout << "Omega" << key << ": "<< endl << OmegaI << endl; 
  //Reconstruct input face image from eigenfaces
  	MatrixXd I_Re = MatrixXd::Zero(HEIGHT*WIDTH, 1);
  	I_Re = FaceSpace * OmegaI;
 	ShowEigenFace("Reconstruct_"+InputImg[key].substr(10), I_Re);
  	
	double d0 = Distance(I_Re, I);
  	if(d0 < 0){
		cout << "Input is not a face." << endl;
  	}
	else{	
		double dist_min = DBL_MAX;
		int index = 0;
		for(int i = 0; i < TrainImg.size(); i++){
		//Find the mimimun distance then classify
			double dist = Distance(OmegaI,  LinearCombines[i]);
			//cout << dist << endl;
			if(dist < dist_min){
				dist_min = dist;
				index = i;
			}
		}
		if(dist_min < T1)
			cout << InputImg[key].substr(10) << " :  dist "<< dist_min  <<" -> It belongs to Eigen Face " << index+1 << "." << endl;  
  		else
			cout << InputImg[key].substr(10) <<  " :  dist " <<  dist_min  << " -> Unknown face." << endl;
	}
  }
  waitKey(0);
  return 0;
}

void SetTrainingImg(set<string> &TrainImg){
  TrainImg.insert("./dataset/subject01.normal.jpg");
  TrainImg.insert("./dataset/subject02.normal.jpg");
  TrainImg.insert("./dataset/subject03.normal.jpg");
  TrainImg.insert("./dataset/subject07.normal.jpg");
  TrainImg.insert("./dataset/subject10.normal.jpg");
  TrainImg.insert("./dataset/subject11.normal.jpg");
  TrainImg.insert("./dataset/subject14.normal.jpg");
  TrainImg.insert("./dataset/subject15.normal.jpg");
  return;
}

void SetInputImg(unordered_map<int, string> &InputImg){
  InputImg.insert(make_pair(0,"./dataset/subject01.normal.jpg"));
  InputImg.insert(make_pair(1,"./dataset/subject01.centerlight.jpg"));
  InputImg.insert(make_pair(2,"./dataset/subject01.happy.jpg"));
  InputImg.insert(make_pair(3,"./dataset/subject02.normal.jpg")); 
  InputImg.insert(make_pair(4,"./dataset/subject03.normal.jpg"));
  InputImg.insert(make_pair(5,"./dataset/subject07.normal.jpg"));
  InputImg.insert(make_pair(6,"./dataset/subject07.centerlight.jpg"));
  InputImg.insert(make_pair(7,"./dataset/subject07.happy.jpg"));
  InputImg.insert(make_pair(8,"./dataset/subject10.normal.jpg"));
  InputImg.insert(make_pair(9,"./dataset/subject11.normal.jpg"));
  InputImg.insert(make_pair(10,"./dataset/subject11.centerlight.jpg"));
  InputImg.insert(make_pair(11,"./dataset/subject11.happy.jpg"));
  InputImg.insert(make_pair(12,"./dataset/subject12.normal.jpg"));
  InputImg.insert(make_pair(13,"./dataset/subject14.normal.jpg"));
  InputImg.insert(make_pair(14,"./dataset/subject14.happy.jpg"));
  InputImg.insert(make_pair(15,"./dataset/subject14.sad.jpg"));
  InputImg.insert(make_pair(16,"./dataset/subject15.normal.jpg"));
  InputImg.insert(make_pair(17,"./dataset/apple1_gray.jpg"));
  return;
}

double Distance(MatrixXd a, MatrixXd b){
	if(a.rows() != b.rows() || a.cols() != b.cols()){
		cout << "Dimesion has to be the same." << endl;
		return -1;
	}
	double res = 0.0;
	for(int i = 0; i < a.rows(); i++){
		for(int j = 0; j < a.cols(); j++){
			res += pow(abs(a(i,j)-b(i,j)), 2);	
		}
	}
	res = sqrt(res);
	return res;
}
void Normalization(MatrixXd &m){
	//Find min and max
	double max_val = DBL_MIN;
	double min_val = DBL_MAX;
	for(int i = 0; i < m.rows(); i++){
		for(int j = 0; j < m.cols(); j++){
			max_val = max(max_val, m(i,j));
			min_val = min(min_val, m(i,j));
		}
	}
	//Normalized
	for(int i = 0; i < m.rows(); i++){
		for(int j = 0; j < m.cols(); j++){
			m(i,j) = (m(i,j) - min_val)/(max_val - min_val) * 255;
		}
	}
	return;
}
void ShowEigenFace(string title, MatrixXd m){
	Mat img = Mat(HEIGHT, WIDTH, CV_8UC1);
	MatrixXd tmp(m.rows(),m.cols());
	tmp = m;
	Normalization(tmp);
	int row = 0;
        for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
			img.at<uchar>(i,j) = tmp(row,0);
			row++;
		}
	}
	imshow(title, img);
	if(title.find(".jpg") == std::string::npos)
		imwrite(title+".jpg", img);
	else
		imwrite(title,img);
	return;
}
MatrixXd Mat_to_Matrix(Mat img){
	MatrixXd mtrx = MatrixXd::Zero(img.rows*img.cols,1);
	int row = 0;
	for(int i = 0; i < img.rows; i++){
		for(int j = 0; j < img.cols; j++){
			mtrx(row,0) = img.at<uchar>(i,j);
			row++;
		}
	} 
	return mtrx;
}
void Insert_Col_MatrixA(MatrixXd &vec, MatrixXd &A, int col){
	for(int i = 0; i < vec.rows(); i++){
		A(i,col) = vec(i,0);
	}	
	return; 
}
