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
			//计算四个方向邻域两像素的差别，也就是欧式距离或者说二阶范数  
			//（当所有像素都算完后，就相当于计算八邻域的像素差了）  
			Vec3d color = img.at<Vec3b>(y, x);
			if (x > 0) // left  >0的判断是为了避免在图像边界的时候还计算，导致越界  
			{
				Vec3d diff = color - (Vec3d)img.at<Vec3b>(y, x - 1);
				beta += diff.dot(diff);  //矩阵的点乘，也就是各个元素平方的和  
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
		beta = 1.f / (2 * beta / (4 * img.cols*img.rows - 3 * img.cols - 3 * img.rows + 2)); //论文公式（5）  

	return beta;
}


void Bilateral::constructGCGraph(const GMM& bgdGMM, const GMM& fgdGMM, GCGraph<double>& graph) {
	double bata = 0.001;
	int vtxCount = calculateVtxCount();  //顶点数，每一个像素是一个顶点  
	int edgeCount = 2 * 64 * vtxCount;  //边数，需要考虑图边界的边的缺失
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
								Vec3f color = gridColor.at<Vec3f>(point);
								double fromSource, toSink;
								double num = grid.at<Vec< int, 2 > >(point)[0];

								fromSource = -log(bgdGMM(color))* sqrt(num);
								toSink = -log(fgdGMM(color))* sqrt(num);
								graph.addTermWeights(vtxIdx, fromSource, toSink);

								//平滑项
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
															double e = exp(-bata*diff.dot(diff));  //矩阵的点乘，也就是各个元素平方的和
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