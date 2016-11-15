#include "Bilateral.h"
#include <iostream>

Bilateral::Bilateral(Mat& img):
	imgSrc(img){}

Bilateral::~Bilateral()
{
}

void Bilateral::InitGmms(std::vector<Point>& forPts, std::vector<Point>& bacPts)
{
	std::vector<Point> m_forePts;      //保存前景点（去重）
	std::vector<Point> m_backPts;      //保存背景点
	std::vector<Vec3f> bgdSamples;    //从背景点存储背景颜色
	std::vector<Vec3f> fgdSamples;    //从前景点存储前景颜色

	//清除重复的点
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

	//对应grid点保存
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



	//添加点的颜色数据
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

	//高斯模型建立
	GMM bgdGMM(bgModel), fgdGMM(fgModel);
	const int kMeansItCount = 10;  //迭代次数  
	const int kMeansType = KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii  
	Mat bgdLabels, fgdLabels; //记录背景和前景的像素样本集中每个像素对应GMM的哪个高斯模型

	//kmeans进行分类
	Mat _bgdSamples((int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0]);
	kmeans(_bgdSamples, GMM::componentsCount, bgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
	Mat _fgdSamples((int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_fgdSamples, GMM::componentsCount, fgdLabels,
		TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);

	//分类后的结果用来训练GMMs（初始化）
	bgdGMM.initLearning();
	for (int i = 0; i < (int)bgdSamples.size(); i++)
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	bgdGMM.endLearning();

	fgdGMM.initLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	fgdGMM.endLearning();

	//迭代训练GMMs模型
	for (int times = 0;times < 3;times++) {
		//times迭代次数
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
			grid.at<Vec2i>(point)[0] += 1;
		}
	}
}

void Bilateral::constructGCGraph(const GMM& bgdGMM, const GMM& fgdGMM, GCGraph<double>& graph) {
	int vtxCount = calculateVtxCount();  //顶点数，每一个像素是一个顶点  
	int edgeCount = 2 * 6 * vtxCount;  //边数，需要考虑图边界的边的缺失
	graph.create(vtxCount, edgeCount);
	
	for (int t = 0; t < gridSize[0]; t++){
		for (int x = 0; x < gridSize[1]; x++){
			for (int y = 0; y < gridSize[2]; y++){
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++){
						for (int b = 0; b < gridSize[5]; b++){
							int point[6] = { t,x,y,r,g,b };

							if (grid.at<Vec2i>(point)[0] > 0) {
								int vtxIdx = graph.addVtx();//存在像素点映射就加顶点

								//先验项
								grid.at<Vec2i>(point)[1] = vtxIdx;
								Vec3b color;//计算grid中顶点对应的颜色
								color[0] = (r * 256 + 256/2) / gridSize[3];//多加0.5是为了把颜色移到方格中心
								color[1] = (g * 256 + 256/2) / gridSize[4];
								color[2] = (b * 256 + 256/2) / gridSize[5];
								double fromSource, toSink;
								fromSource = -log(bgdGMM(color));
								toSink = -log(fgdGMM(color));
								graph.addTermWeights(vtxIdx, fromSource, toSink);

								//平滑项
								if (t > 0) {
									int pointN[6] = { t-1,x,y,r,g,b };
									if (grid.at<Vec2i>(pointN)[0] > 0) {
										double w = 0.5 * grid.at<Vec2i>(point)[0] * grid.at<Vec2i>(pointN)[0];
										w = sqrt(w);
										graph.addEdges(vtxIdx, grid.at<Vec2i>(pointN)[1], w, w);
									}
								}
								if (x > 0) {
									int pointN[6] = { t,x-1,y,r,g,b };
									if (grid.at<Vec2i>(pointN)[0] > 0) {
										double w = 0.5 * grid.at<Vec2i>(point)[0] * grid.at<Vec2i>(pointN)[0];
										w = sqrt(w);
										graph.addEdges(vtxIdx, grid.at<Vec2i>(pointN)[1], w, w);
									}
								}
								if (y > 0) {
									int pointN[6] = { t,x,y - 1,r,g,b };
									if (grid.at<Vec2i>(pointN)[0] > 0) {
										double w = 0.5 * grid.at<Vec2i>(point)[0] * grid.at<Vec2i>(pointN)[0];
										w = sqrt(w);
										graph.addEdges(vtxIdx, grid.at<Vec2i>(pointN)[1], w, w);
									}
								}
								if (r > 0) {
									int pointN[6] = { t,x,y,r-1,g,b };
									if (grid.at<Vec2i>(pointN)[0] > 0) {
										double w = 0.2 * grid.at<Vec2i>(point)[0] * grid.at<Vec2i>(pointN)[0];
										w = sqrt(w);
										graph.addEdges(vtxIdx, grid.at<Vec2i>(pointN)[1], w, w);
									}
								}								
								if (g > 0) {
									int pointN[6] = { t,x,y,r,g-1,b };
									if (grid.at<Vec2i>(pointN)[0] > 0) {
										double w = 0.2 * grid.at<Vec2i>(point)[0] * grid.at<Vec2i>(pointN)[0];
										w = sqrt(w);
										graph.addEdges(vtxIdx, grid.at<Vec2i>(pointN)[1], w, w);
									}
								}
								if (b > 0) {
									int pointN[6] = { t,x,y,r,g,b-1 };
									if (grid.at<Vec2i>(pointN)[0] > 0) {
										double w = 0.2 * grid.at<Vec2i>(point)[0] * grid.at<Vec2i>(pointN)[0];
										w = sqrt(w);
										graph.addEdges(vtxIdx, grid.at<Vec2i>(pointN)[1], w, w);
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
	graph.maxFlow();//最大流图割

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
	point[2] = gridSize[2] * p.x / imgSrc.cols;//x,y互换、由于p坐标存错，导致的问题。
	Vec3b color = (Vec3b)imgSrc.at<Vec3b>(p);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}

void Bilateral::run(Mat& mask) {
	GMM bgdGMM(bgModel), fgdGMM(fgModel);//前背景模型
	GCGraph<double> graph;//图割
	mask = Mat::zeros(imgSrc.rows, imgSrc.cols, CV_8UC1);
	initGrid();
	constructGCGraph(bgdGMM, fgdGMM, graph);
	estimateSegmentation(graph, mask);
}