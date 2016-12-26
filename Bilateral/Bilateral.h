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
	Mat grid, gridColor;	//升维，平均取点，得到的grid。6维数组，保存顶点值与邻近像素点总数。
	const int gridSize[6] = { 1,20,30,16,16,16 };	//grid各个维度的大小,按顺序来为：t,x,y,r,g,b。
	std::vector<std::vector<int> > grid_forePts;      //前景grid点
	std::vector<std::vector<int> > grid_backPts;      //背景grid点
public:
	Bilateral(Mat& img);
	~Bilateral();
	void InitGmms(std::vector<Point>& forPts, std::vector<Point>& bacPts);
	void run(Mat& );
private:
	bool isPtInVector(Point pt, std::vector<Point>& points);
	void initGrid();
	void constructGCGraph(const GMM&, const GMM&, GCGraph<double>& graph);
	int calculateVtxCount();
	void estimateSegmentation(GCGraph<double>&, Mat&);
	void getGridPoint(const Point , int *);
	void getGridPoint(const Point , std::vector<int>& );
};

#endif