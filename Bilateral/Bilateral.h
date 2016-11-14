#pragma once
#include "opencv2/imgproc.hpp"
#include "GMM.h"
#include "gcgraph.hpp"

#ifndef _BILATERAL_H_
#define _BILATERAL_H_

using namespace cv;

class Bilateral
{
public:
	Mat imgSrc;		//输入图片数据
	Mat bgModel, fgModel;	//前背景高斯模型
	Mat grid;	//升维，平均取点，得到的grid。6维数组，保存顶点值与邻近像素点总数。
	const int gridSize[6] = { 1,40,40,32,32,32 };	//grid各个维度的大小,按顺序来为：t,x,y,r,g,b。
public:
	Bilateral(Mat& img);
	~Bilateral();
	void InitGmms(std::vector<Point>& forPts, std::vector<Point>& bacPts);
	void run();
private:
	bool isPtInVector(Point pt, std::vector<Point>& points);
	void initGrid();
};

#endif