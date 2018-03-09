#pragma once
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "GMM.h"
#include "gcgraph.hpp"

#ifndef _BILATERAL_H_
#define _BILATERAL_H_

using namespace cv;

enum girdIndex {
	pixSum = 0,  //���ص���
	fgdSum = 1,  //ǰ���������ڽ���ֵ��
	bgdSum = 2,  //����
	vIdx = 3,   //�����ǩ
};

class Bilateral
{
public:
	Mat imgSrc;	 //����ͼƬ����
	Mat bgModel, fgModel, unModel;	//ǰ������˹ģ��
	Mat grid,gridColor, gridProbable;	//��ά��ƽ��ȡ�㣬�õ���grid��6ά���飬���涥��ֵ���ڽ����ص�������
	bool haveUnModel;//unModel�Ƿ����
	int gridSize[6] = { 1,20,30,16,16,16 };	//grid����ά�ȵĴ�С,��˳����Ϊ��t,x,y,r,g,b��
	//int gridSize[6] = { 1,40,50,1,1,1 };	
public:
	Bilateral(Mat img);
	~Bilateral();
	void InitGmms(Mat& );
	void run(Mat& );
	void getGmmProMask(Mat& mask);
	void savePreImg(std::string path, GCGraph<double>& graph);
private:
	void initGrid();
	void constructGCGraph(GCGraph<double>& graph);
	int calculateVtxCount();
	void estimateSegmentation(GCGraph<double>&, Mat& );
	void getGridPoint(int , const Point , int *, int , int , int );
	void getGridPoint(int , const Point , std::vector<int>& , int , int , int );
	void getColor();
};

#endif