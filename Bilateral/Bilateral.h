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
	Mat imgSrc;		//����ͼƬ����
	Mat bgModel, fgModel;	//ǰ������˹ģ��
	Mat grid, gridColor;	//��ά��ƽ��ȡ�㣬�õ���grid��6ά���飬���涥��ֵ���ڽ����ص�������
	const int gridSize[6] = { 1,20,30,16,16,16 };	//grid����ά�ȵĴ�С,��˳����Ϊ��t,x,y,r,g,b��
	std::vector<std::vector<int> > grid_forePts;      //ǰ��grid��
	std::vector<std::vector<int> > grid_backPts;      //����grid��
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