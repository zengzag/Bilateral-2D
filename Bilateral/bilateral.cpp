#include "Bilateral.h"
#include <iostream>

Bilateral::Bilateral(Mat& img):
	imgSrc(img){}

Bilateral::~Bilateral()
{
}

void Bilateral::InitGmms(std::vector<Point>& forPts, std::vector<Point>& bacPts)
{
	std::vector<Point> m_forePts;      //����ǰ���㣨ȥ�أ�
	std::vector<Point> m_backPts;      //���汳����
	std::vector<Vec3f> bgdSamples;    //�ӱ�����洢������ɫ
	std::vector<Vec3f> fgdSamples;    //��ǰ����洢ǰ����ɫ

	//����ظ��ĵ�
	m_forePts.clear();
	for (int i = 0;i<forPts.size();i++)
	{
		if (!isPtInVector(forPts[i], m_forePts))
			m_forePts.push_back(forPts[i]);
	}
	m_backPts.clear();
	for (int i = 0;i<bacPts.size();i++)
	{
		if (!isPtInVector(bacPts[i], m_backPts))
			m_backPts.push_back(bacPts[i]);
	}

	//��Ӧgrid�㱣��
	for (int i = 0;i<m_forePts.size();i++)
	{
		std::vector<int> gridPoint(6);
		getGridPoint(m_forePts[i], gridPoint);
		grid_forePts.push_back(gridPoint);
	}
	for (int i = 0;i<m_backPts.size();i++)
	{
		std::vector<int> gridPoint(6);
		getGridPoint(m_backPts[i], gridPoint);
		grid_backPts.push_back(gridPoint);
	}



	//��ӵ����ɫ����
	for (int i = 0;i<m_forePts.size();i++)
	{
		Vec3f color = (Vec3f)imgSrc.at<Vec3b>(m_forePts[i]);
		bgdSamples.push_back(color);
	}
	for (int i = 0;i<m_backPts.size();i++)
	{
		Vec3f color = (Vec3f)imgSrc.at<Vec3b>(m_backPts[i]);
		fgdSamples.push_back(color);
	}

	//��˹ģ�ͽ���
	GMM bgdGMM(bgModel), fgdGMM(fgModel);
	const int kMeansItCount = 10;  //��������  
	const int kMeansType = KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii  
	Mat bgdLabels, fgdLabels; //��¼������ǰ����������������ÿ�����ض�ӦGMM���ĸ���˹ģ��

	//kmeans���з���
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	kmeans(_bgdSamples, GMM::componentsCount, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, GMM::componentsCount, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

	//�����Ľ������ѵ��GMMs����ʼ����
	bgdGMM.initLearning();
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.endLearning();

	fgdGMM.initLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.endLearning();

	//����ѵ��GMMsģ��
	for (int times = 0;times < 3;times++) {
		//times��������
		for (int i = 0; i < (int)bgdSamples.size(); i++) {
			Vec3d color = bgdSamples[i];
			bgdLabels.at<int>(i, 0) = bgdGMM.whichComponent(color);
		}
		for (int i = 0; i < (int)fgdSamples.size(); i++) {
			Vec3d color = fgdSamples[i];
			fgdLabels.at<int>(i, 0) = fgdGMM.whichComponent(color);
		}
		bgdGMM.initLearning();
		for (int i = 0; i < (int)bgdSamples.size(); i++)
			bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
		bgdGMM.endLearning();

		fgdGMM.initLearning();
		for (int i = 0; i < (int)fgdSamples.size(); i++)
			fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
		fgdGMM.endLearning();
	}
}

bool Bilateral::isPtInVector(Point pt, std::vector<Point>& points)
{
	for (int i = 0;i<points.size();i++) {
		if (pt.x == points[i].x&&pt.y == points[i].y) {
			return true;
		}
	}
	return false;
}

void Bilateral::initGrid() {
	Mat L(6, gridSize, CV_32SC(2), Scalar::all(0));
	grid = L;
	Mat C(6, gridSize, CV_32FC(3), Scalar::all(0));
	gridColor = C;
	int xSize = imgSrc.rows;
	int ySize = imgSrc.cols;
	for (int x = 0; x < xSize; x++)
	{
		for (int y = 0; y < ySize; y++)
		{
			int xNew = gridSize[1] * x / xSize ;
			int yNew = gridSize[2] * y / ySize ;
			Vec3b color = (Vec3b)imgSrc.at<Vec3b>(x,y);
			int rNew = gridSize[3] * color[0] / 256;
			int gNew = gridSize[4] * color[1] / 256;
			int bNew = gridSize[5] * color[2] / 256;
			int point[6] = { 0,xNew,yNew,rNew,gNew,bNew };
			int count = ++(grid.at<Vec2i>(point)[0]);
			Vec3f colorMeans = gridColor.at<Vec3f>(point);
			colorMeans[0] = colorMeans[0] * (count - 1.0) / (count + 0.0) + color[0] / (count + 0.0);
			colorMeans[1] = colorMeans[1] * (count - 1.0) / (count + 0.0) + color[1] / (count + 0.0);
			colorMeans[2] = colorMeans[2] * (count - 1.0) / (count + 0.0) + color[2] / (count + 0.0);
			gridColor.at<Vec3f>(point) = colorMeans;
		}
	}
}


static double calcBeta(const Mat& img)
{
	double beta = 0;
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			//�����ĸ��������������صĲ��Ҳ����ŷʽ�������˵���׷���  
			//�����������ض�����󣬾��൱�ڼ������������ز��ˣ�  
			Vec3d color = img.at<Vec3b>(y, x);
			if (x > 0) // left  >0���ж���Ϊ�˱�����ͼ��߽��ʱ�򻹼��㣬����Խ��  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
				beta += diff.dot(diff);  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�  
			}
			if (y > 0 && x > 0) // upleft  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x - 1);
				beta += diff.dot(diff);
			}
			if (y > 0) // up  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x);
				beta += diff.dot(diff);
			}
			if (y > 0 && x < img.cols - 1) // upright  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y - 1, x + 1);
				beta += diff.dot(diff);
			}
		}
	}
	if (beta <= std::numeric_limits<double>::epsilon())
		beta = 0;
	else
		beta = 1.f / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2)); //���Ĺ�ʽ��5��  

	return beta;
}


void Bilateral::constructGCGraph(const GMM& bgdGMM, const GMM& fgdGMM, GCGraph<double>& graph) {
	double bata = 0.001;
	int vtxCount = calculateVtxCount();  //��������ÿһ��������һ������  
	int edgeCount = 2 * 64 * vtxCount;  //��������Ҫ����ͼ�߽�ıߵ�ȱʧ
	graph.create(vtxCount, edgeCount);
	
	for (int t = 0; t < gridSize[0]; t++){
		for (int x = 0; x < gridSize[1]; x++){
			for (int y = 0; y < gridSize[2]; y++){
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++){
						for (int b = 0; b < gridSize[5]; b++){
							int point[6] = { t,x,y,r,g,b };

							if (grid.at<Vec2i>(point)[0] > 0) {
								int vtxIdx = graph.addVtx();//�������ص�ӳ��ͼӶ���

								//������
								grid.at<Vec2i>(point)[1] = vtxIdx;
								Vec3f color = gridColor.at<Vec3f>(point);
								double fromSource, toSink;
								double num = grid.at<Vec< int, 2 > >(point)[0];

								fromSource = -log(bgdGMM(color))* sqrt(num);
								toSink = -log(fgdGMM(color))* sqrt(num);
								graph.addTermWeights(vtxIdx, fromSource, toSink);

								//ƽ����
								int count = 0;
								for (int tN = t; tN > t - 2 && tN >= 0; tN--) {
									for (int xN = x; xN > x - 2 && xN >= 0; xN--) {
										for (int yN = y; yN > y - 2 && yN >= 0; yN--) {
											for (int rN = r; rN > r - 2 && rN >= 0; rN--) {
												for (int gN = g; gN > g - 2 && gN >= 0; gN--) {
													for (int bN = b; bN > b - 2 && bN >= 0; bN--) {
														int pointN[6] = { tN,xN,yN,rN,gN,bN };
														count++;
														if (grid.at<Vec< int, 2 > >(pointN)[0] > 0 && count > 1) {
															double num = sqrt(grid.at<Vec< int, 2 > >(point)[0] * grid.at<Vec< int, 2 > >(pointN)[0] + 1);
															Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
															double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
															double w = e * sqrt(num);
															graph.addEdges(vtxIdx, grid.at<Vec< int, 2 > >(pointN)[1], w, w);

														}
													}
												}
											}
										}
									}
								}

							}
						}
					}
				}

			}
		}
	}

	for (int i = 0;i < grid_forePts.size();i++) {
		int point[6] = { 0,0,0,0,0,0 };
		point[0] = grid_forePts[i][0];
		point[1] = grid_forePts[i][1];
		point[2] = grid_forePts[i][2];
		point[3] = grid_forePts[i][3];
		point[4] = grid_forePts[i][4];
		point[5] = grid_forePts[i][5];
		if (grid.at<Vec2i>(point)[1] != 0) {
			graph.addTermWeights(grid.at<Vec2i>(point)[1], 0, 9999);
		}
	}
	for (int i = 0;i < grid_backPts.size();i++) {
		int point[6] = { 0,0,0,0,0,0 };
		point[0] = grid_backPts[i][0];
		point[1] = grid_backPts[i][1];
		point[2] = grid_backPts[i][2];
		point[3] = grid_backPts[i][3];
		point[4] = grid_backPts[i][4];
		point[5] = grid_backPts[i][5];
		if (grid.at<Vec2i>(point)[1] != 0) {
			graph.addTermWeights(grid.at<Vec2i>(point)[1], 9999, 0);
		}
	}

}


int Bilateral::calculateVtxCount() {
	int count=0;
	for (int t = 0; t < gridSize[0]; t++)
	{
		for (int x = 0; x < gridSize[1]; x++)
		{
			for (int y = 0; y < gridSize[2]; y++)
			{
				for (int r = 0; r < gridSize[3]; r++)
				{
					for (int g = 0; g < gridSize[4]; g++)
					{
						for (int b = 0; b < gridSize[5]; b++)
						{
							int point[6] = { t,x,y,r,g,b };
							if (grid.at<Vec2i>(point)[0] > 0) {
								count++;
							}
						}
					}
				}
			}
		}
	}
	return count;
}

void Bilateral::estimateSegmentation(GCGraph<double>& graph, Mat& mask) {
	graph.maxFlow();//�����ͼ��

	Point p;
	for (p.y = 0; p.y < mask.cols; p.y++)
	{
		for (p.x = 0; p.x < mask.rows; p.x++)
		{

			int point[6] = {0,0,0,0,0,0};
			getGridPoint(p, point);
			int vertex = grid.at<Vec2i>(point)[1];
			if (graph.inSourceSegment(vertex))
				mask.at<uchar>(p.x,p.y) = 0;
			else
				mask.at<uchar>(p.x, p.y) = 1;
		}
	}

}

void Bilateral::getGridPoint(const Point p,int *point) {
	point[0] = 0;
	point[1] = gridSize[1] * p.x / imgSrc.rows;
	point[2] = gridSize[2] * p.y / imgSrc.cols;
	Vec3b color = (Vec3b)imgSrc.at<Vec3b>(p.x, p.y);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}

void Bilateral::getGridPoint(const Point p, std::vector<int>& point) {
	point[0] = 0;
	point[1] = gridSize[1] * p.y / imgSrc.rows;
	point[2] = gridSize[2] * p.x / imgSrc.cols;//x,y����������p���������µ����⡣
	Vec3b color = (Vec3b)imgSrc.at<Vec3b>(p);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}

void Bilateral::run(Mat& mask) {
	GMM bgdGMM(bgModel), fgdGMM(fgModel);//ǰ����ģ��
	GCGraph<double> graph;//ͼ��
	mask = Mat::zeros(imgSrc.rows, imgSrc.cols, CV_8UC1);
	initGrid();
	constructGCGraph(bgdGMM, fgdGMM, graph);
	estimateSegmentation(graph, mask);
}