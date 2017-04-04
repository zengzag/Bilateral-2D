#include "Bilateral.h"
#include <iostream>
#include <fstream>


Bilateral::Bilateral(Mat img) :
	imgSrc(img) {
	gridSize[1] = img.rows;
	gridSize[2] = img.cols;
	initGrid();
}

Bilateral::~Bilateral()
{
	grid.release();
	gridColor.release();
}

void Bilateral::InitGmms(Mat& mask)
{
	double _time = static_cast<double>(getTickCount());//��ʱ


	int tSize = 1;
	int xSize = imgSrc.rows;
	int ySize = imgSrc.cols;
	int point[6] = { 0,0,0,0,0,0 };

	std::vector<Vec3f> bgdSamples;    //�ӱ�����洢������ɫ
	std::vector<Vec3f> fgdSamples;    //��ǰ����洢ǰ����ɫ


	for (int x = 0; x < xSize; x++)
	{
		for (int y = 0; y < ySize; y++)
		{
			uchar a = mask.at<uchar>(x, y);
			if (mask.at<uchar>(x, y) == GC_BGD) {
				getGridPoint(0, Point(x, y), point, tSize, xSize, ySize);
				grid.at<Vec< int, 4 > >(point)[bgdSum] += 3;
				/*Vec3b color = imgSrc.at<Vec3b>(x,y);
				bgdSamples.push_back(color);*/
			}
			else if (mask.at<uchar>(x, y) == GC_FGD/*GC_FGD*/) {
				getGridPoint(0, Point(x, y), point, tSize, xSize, ySize);
				grid.at<Vec< int, 4 > >(point)[fgdSum] += 3;
				/*Vec3b color = imgSrc.at<Vec3b>(x, y);
				fgdSamples.push_back(color);*/
			}
			else if (mask.at<uchar>(x, y) == GC_PR_FGD/*GC_FGD*/) {
				getGridPoint(0, Point(x, y), point, tSize, xSize, ySize);
				grid.at<Vec< int, 4 > >(point)[fgdSum] += 1;
				/*Vec3b color = imgSrc.at<Vec3b>(x, y);
				fgdSamples.push_back(color);*/
			}
			else if (mask.at<uchar>(x, y) == GC_PR_BGD/*GC_FGD*/) {
				getGridPoint(0, Point(x, y), point, tSize, xSize, ySize);
				grid.at<Vec< int, 4 > >(point)[bgdSum] += 1;
				/*Vec3b color = imgSrc.at<Vec3b>(x, y);
				bgdSamples.push_back(color);*/
			}
		}
	}

	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int pixCount = grid.at<Vec< int, 4 > >(point)[pixSum];
							if (pixCount > 0) {
								int bgdcount = grid.at<Vec< int, 4 > >(point)[bgdSum];
								int fgdcount = grid.at<Vec< int, 4 > >(point)[fgdSum];
								if (bgdcount > 0) {
									Vec3f color = gridColor.at<Vec3f>(point);
									bgdSamples.push_back(color);
								}

								if (fgdcount > 0) {
									Vec3f color = gridColor.at<Vec3f>(point);
									fgdSamples.push_back(color);
								}
							}
						}
					}
				}
			}
		}
	}


	/*std::vector<double> data(256,0.0);
	std::ofstream f1("E:/Projects/OpenCV/DAVIS-data/examples/output/color.txt");
	if (!f1)return;
	int sunbgd = bgdSamples.size();
	for (int i = sunbgd - 1; i >= 0 ; i--) {
		Vec3f color = bgdSamples[i];
		data[(int)color[0]]++;
	}
	for (int i = 0; i < 256; i++) {
		f1 << data[i]/ sunbgd << std::endl;
	}
	f1.close();*/

	std::ofstream f1("E:/Projects/OpenCV/DAVIS-data/examples/output/color2.txt");
	if (!f1)return;
	for (int i = bgdSamples.size() - 1; i >= 0; i--) {
		Vec3f color = bgdSamples[i];
		f1 << color[0] << std::endl;
	}
	f1.close();




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
		bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i], 1);
	bgdGMM.endLearning();
	fgdGMM.initLearning();
	for (int i = 0; i < (int)fgdSamples.size(); i++)
		fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i], 1);
	fgdGMM.endLearning();
	for (int times = 0; times < 3; times++)
	{
		//ѵ��GMMsģ��
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
			bgdGMM.addSample(bgdLabels.at<int>(i, 0), bgdSamples[i], 1);
		bgdGMM.endLearning();
		fgdGMM.initLearning();
		for (int i = 0; i < (int)fgdSamples.size(); i++)
			fgdGMM.addSample(fgdLabels.at<int>(i, 0), fgdSamples[i], 1);
		fgdGMM.endLearning();
	}


	std::vector<Vec3f> unSamples;    //��������

	for (int i = 0; i < (int)bgdSamples.size(); i++) {
		Vec3d color = bgdSamples[i];
		double b = bgdGMM(color), f = fgdGMM(color);
		if (fgdGMM.bigThan1Cov(color) || (b < f)) {
			unSamples.push_back(color);
		}
	}
	for (int i = 0; i < (int)fgdSamples.size(); i++) {
		Vec3d color = fgdSamples[i];
		double b = bgdGMM(color), f = fgdGMM(color);
		if (bgdGMM.bigThan1Cov(color) || (b > f)) {
			unSamples.push_back(color);
		}
	}
	if (unSamples.size() < 10) {
		haveUnModel = false;
	}
	else {
		haveUnModel = true;
		GMM unGMM(unModel);
		Mat unLabels;
		Mat _unSamples((int)unSamples.size(), 3, CV_32FC1, &unSamples[0][0]);
		kmeans(_unSamples, GMM::componentsCount, unLabels,
			TermCriteria(CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType);
		unGMM.initLearning();
		for (int i = 0; i < (int)unSamples.size(); i++)
			unGMM.addSample(unLabels.at<int>(i, 0), unSamples[i], 1);
		unGMM.endLearning();
		for (int times = 0; times < 3; times++)
		{
			//ѵ��GMMsģ��
			for (int i = 0; i < (int)unSamples.size(); i++) {
				Vec3d color = unSamples[i];
				unLabels.at<int>(i, 0) = unGMM.whichComponent(color);
			}
			unGMM.initLearning();
			for (int i = 0; i < (int)unSamples.size(); i++)
				unGMM.addSample(unLabels.at<int>(i, 0), unSamples[i], 1);
			unGMM.endLearning();
		}

	}


	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("��˹��ģ��ʱ%f\n", _time);//��ʾʱ��
}

void Bilateral::initGrid() {
	double _time = static_cast<double>(getTickCount());

	Mat L(6, gridSize, CV_32SC(4), Scalar(0, 0, 0, -1));
	Mat C(6, gridSize, CV_32FC(3), Scalar::all(0));
	Mat P(6, gridSize, CV_32FC(3), Scalar::all(0));
	grid = L;gridColor = C;gridProbable = P;
	int tSize = 1;
	int xSize = imgSrc.rows;
	int ySize = imgSrc.cols;
	for (int t = 0; t < tSize; t++)
	{
		for (int x = 0; x < xSize; x++)
		{
			//#pragma omp parallel for
			for (int y = 0; y < ySize; y++)
			{
				int tNew = gridSize[0] * t / tSize;
				int xNew = gridSize[1] * x / xSize;
				int yNew = gridSize[2] * y / ySize;
				Vec3b color = (Vec3b)imgSrc.at<Vec3b>(x, y);
				int rNew = gridSize[3] * color[0] / 256;
				int gNew = gridSize[4] * color[1] / 256;
				int bNew = gridSize[5] * color[2] / 256;
				int point[6] = { tNew,xNew,yNew,rNew,gNew,bNew };
				int count = ++(grid.at<Vec< int, 4 > >(point)[pixSum]);
				Vec3f colorMeans = gridColor.at<Vec3f>(point);
				colorMeans[0] = colorMeans[0] * (count - 1.0) / (count + 0.0) + color[0] / (count + 0.0);
				colorMeans[1] = colorMeans[1] * (count - 1.0) / (count + 0.0) + color[1] / (count + 0.0);
				colorMeans[2] = colorMeans[2] * (count - 1.0) / (count + 0.0) + color[2] / (count + 0.0);
				gridColor.at<Vec3f>(point) = colorMeans;
			}
		}
	}

	

	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("����grid��ʱ%f\n", _time);//��ʾʱ��
}

void Bilateral::savePreImg(std::string path,GCGraph<double>& graph) {
	int tSize = 1;
	int xSize = imgSrc.rows;
	int ySize = imgSrc.cols;
	Mat randColor(6, gridSize, CV_8UC3, Scalar::all(0));
	
	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							if (grid.at<Vec< int, 4 > >(point)[pixSum] > 0) {
								int vertex = grid.at<Vec< int, 4 > >(point)[vIdx];
								if (graph.inSourceSegment(vertex))
									randColor.at<Vec3b>(point) = { (uchar)(rand() % 200),(uchar)(rand() % 200),(uchar)(rand() % 200) };
								else
									randColor.at<Vec3b>(point) = { (uchar)(rand() % 155+100),(uchar)(rand() % 155 + 100),(uchar)(rand() % 155 + 100) };
							}
						}
					}
				}
			}
		}
	}

	Mat preSegImg(imgSrc.rows, imgSrc.cols, CV_8UC3);
	for (int y = 0; y < ySize; y++)
	{
		for (int x = 0; x < xSize; x++)
		{
			Point p(x, y);
			int point[6] = { 0,0,0,0,0,0 };
			getGridPoint(0, p, point, tSize, xSize, ySize);
			Vec3b colorMeans = randColor.at<Vec3b>(point);
			preSegImg.at<Vec3b>(x, y) = colorMeans;
		}
	}
	randColor.release();
	imwrite(path, preSegImg);
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

void Bilateral::constructGCGraph(GCGraph<double>& graph) {
	double _time = static_cast<double>(getTickCount());

	double bata = calcBeta(imgSrc);
	int vtxCount = calculateVtxCount();  //��������ÿһ��������һ������  
	int edgeCount = 2 * 256 * vtxCount;  //��������Ҫ����ͼ�߽�ıߵ�ȱʧ
	graph.create(vtxCount, edgeCount);
	int eCount = 0, eCount2 = 0, eCount3 = 0;
	GMM bgdGMM(bgModel), fgdGMM(fgModel), unGMM(unModel);
	bgdGMM.save();
	for (int t = 0; t < gridSize[0]; t++) {
		for (int x = 0; x < gridSize[1]; x++) {
			for (int y = 0; y < gridSize[2]; y++) {
				for (int r = 0; r < gridSize[3]; r++) {
					for (int g = 0; g < gridSize[4]; g++) {
						for (int b = 0; b < gridSize[5]; b++) {
							int point[6] = { t,x,y,r,g,b };
							int pixCount = grid.at<Vec< int, 4 > >(point)[pixSum];
							if (pixCount > 0) {
								int vtxIdx = graph.addVtx();//�������ص�ӳ��ͼӶ���								
								//������
								grid.at<Vec< int, 4 > >(point)[vIdx] = vtxIdx;
								Vec3f color = gridColor.at<Vec3f>(point);
								double fromSource, toSink;
								double fSum = grid.at<Vec< int, 4 > >(point)[fgdSum];
								double bSum = grid.at<Vec< int, 4 > >(point)[bgdSum];

								//�ۺϷ���
								if ((bSum > 0) && fSum == 0) {
									fromSource = 0;
									toSink = 9999;
									gridProbable.at<Vec3f>(point)[0] = 0;
								}
								else if (bSum == 0 && (fSum > 0)) {
									fromSource = 9999;
									toSink = 0;
									gridProbable.at<Vec3f>(point)[0] = 1;
								}
								else {
									double bgd = bgdGMM(color);
									double fgd = fgdGMM(color);
									double un;//��ɫģ�͵Ŀ��Ŷ�,Խ��Խ�����š�
									if (haveUnModel)
										un = unGMM(color);
									else
										un = 0;

									if ((unGMM.bigThan1Cov(color)) || (bgdGMM.smallThan2Cov(color) && fgdGMM.smallThan2Cov(color))) {
										bgd = fgd;
										eCount3++;
									}

									gridProbable.at<Vec3f>(point)[0] = fgd / (bgd + fgd);

									fromSource = -log(bgd)*sqrt(pixCount);
									toSink = -log(fgd)*sqrt(pixCount);
								}
								graph.addTermWeights(vtxIdx, fromSource, toSink);

								//ƽ����
								for (int tN = t; tN > t - 2 && tN >= 0 && tN < gridSize[0]; tN--) {
									for (int xN = x; xN > x - 2 && xN >= 0 && xN < gridSize[1]; xN--) {
										for (int yN = y; yN > y - 2 && yN >= 0 && yN < gridSize[2]; yN--) {
											for (int rN = r; rN > r - 2 && rN >= 0 && rN < gridSize[3]; rN--) {
												for (int gN = g; gN > g - 2 && gN >= 0 && gN < gridSize[4]; gN--) {
													for (int bN = b; bN > b - 2 && bN >= 0 && bN < gridSize[5]; bN--) {
														int pointN[6] = { tN,xN,yN,rN,gN,bN };
														int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
														if (grid.at<Vec< int, 4 > >(pointN)[pixSum] > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
															double num = sqrt(grid.at<Vec< int, 4 > >(point)[pixSum] * grid.at<Vec< int, 4 > >(pointN)[pixSum] + 1);
															Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
															double e = exp(-bata*diff.dot(diff));  //����ĵ�ˣ�Ҳ���Ǹ���Ԫ��ƽ���ĺ�
															double w = 100.0 * e * sqrt(num);
															graph.addEdges(vtxIdx, vtxIdxNew, w, w);
															eCount++;
														}
													}
												}
											}
										}
									}
								}


								/*for (int tN = t; tN >= 0 && tN > t - 2;tN--) {
									for (int xN = 0; xN < x; xN++) {
										for (int yN = 0; yN < gridSize[2]; yN++) {
											int pointN[6] = { tN,xN,yN,r,g,b };
											int vtxIdxNew = grid.at<Vec< int, 4 > >(pointN)[vIdx];
											int vNewPixCount = grid.at<Vec< int, 4 > >(pointN)[pixSum];

											if (vNewPixCount > 0 && vtxIdxNew > 0 && vtxIdxNew != vtxIdx) {
												double vPixSumDiff = (double)pixCount / (double)vNewPixCount;

												Vec3d diff = (Vec3d)color - (Vec3d)gridColor.at<Vec3f>(pointN);
												double colorDst = (diff.dot(diff));
												if (vPixSumDiff > 0.5 && vPixSumDiff < 2 && colorDst < 128.0) {
													double w = 0.4 * exp(-bata*colorDst) * sqrt(vNewPixCount);
													graph.addEdges(vtxIdx, vtxIdxNew, w, w);
													eCount++;
													eCount2++;
												};
											}
										}
									}
								}*/


							}
						}
					}
				}

			}
		}
	}


	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("ͼ�ͼ��ʱ %f\n", _time);//��ʾʱ��
	printf("��������� %d\n", vtxCount);
	printf("�ߵ����� %d\n", eCount);
	printf("e3������ %d\n", eCount2);
	printf("unWeight<0.5������ %d\n", eCount3);
}


int Bilateral::calculateVtxCount() {
	int count = 0;
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
							if (grid.at<Vec< int, 4 > >(point)[pixSum] > 0) {
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
	double _time = static_cast<double>(getTickCount());
	graph.maxFlow();//�����ͼ��
	_time = (static_cast<double>(getTickCount()) - _time) / getTickFrequency();
	printf("ͼ��ָ���ʱ %f\n", _time);//��ʾʱ��

	double _time2 = static_cast<double>(getTickCount());
	int tSize = 1;
	int xSize = imgSrc.rows;
	int ySize = imgSrc.cols;
	for (int t = 0; t < tSize; t++)
	{
		for (int y = 0; y < ySize; y++)
		{
			for (int x = 0; x < xSize; x++)
			{
				Point p(x, y);
				int point[6] = { 0,0,0,0,0,0 };
				getGridPoint(t, p, point, tSize, xSize, ySize);
				int vertex = grid.at<Vec< int, 4 > >(point)[vIdx];
				if (graph.inSourceSegment(vertex))
					mask.at<uchar>(p.x, p.y) = 1;
				else
					mask.at<uchar>(p.x, p.y) = 0;
			}
		}
	}

	_time2 = (static_cast<double>(getTickCount()) - _time2) / getTickFrequency();
	printf("grid�������mask��ʱ %f\n", _time2);//��ʾʱ��
}

void Bilateral::getGridPoint(int index, const Point p, int *point, int tSize, int xSize, int ySize) {
	point[0] = gridSize[0] * index / tSize;
	point[1] = gridSize[1] * p.x / xSize;
	point[2] = gridSize[2] * p.y / ySize;
	Vec3b color = (Vec3b)imgSrc.at<Vec3b>(p.x, p.y);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}

void Bilateral::getGridPoint(int index, const Point p, std::vector<int>& point, int tSize, int xSize, int ySize) {
	point[0] = gridSize[0] * index / tSize;
	point[1] = gridSize[1] * p.y / xSize;
	point[2] = gridSize[2] * p.x / ySize;//x,y����������p���������µ����⡣
	Vec3b color = (Vec3b)imgSrc.at<Vec3b>(p);
	point[3] = gridSize[3] * color[0] / 256;
	point[4] = gridSize[4] * color[1] / 256;
	point[5] = gridSize[5] * color[2] / 256;
}


void Bilateral::getColor() {
	std::ofstream f1("E:/Projects/OpenCV/DAVIS-data/image/1color.txt");
	if (!f1)return;

	for (int r = 0; r < gridSize[3]; r++) {
		for (int g = 0; g < gridSize[4]; g++) {
			for (int b = 0; b < gridSize[5]; b++) {
				f1 << "---------------------------------" << std::endl;
				Vec3b color;//����grid�ж����Ӧ����ɫ
				color[0] = (r * 256 + 256 / 2) / gridSize[3];//���256/2��Ϊ�˰���ɫ�Ƶ���������
				color[1] = (g * 256 + 256 / 2) / gridSize[4];
				color[2] = (b * 256 + 256 / 2) / gridSize[5];
				f1 << (int)color[0] << "\t" << (int)color[1] << "\t" << (int)color[2] << std::endl;

				for (int t = 0; t < gridSize[0]; t++) {
					for (int x = 0; x < gridSize[1]; x++) {
						for (int y = 0; y < gridSize[2]; y++) {

							int point[6] = { t,x,y,r,g,b };
							if (grid.at<Vec< int, 4 > >(point)[pixSum] != -1) {
								Vec3f colorM = gridColor.at<Vec3f>(point);
								f1 << (float)colorM[0] << "\t" << (float)colorM[1] << "\t" << (float)colorM[2] << std::endl;

							}
						}
					}
				}
			}
		}
	}
	f1.close();
}


void Bilateral::getGmmProMask(Mat& mask) {
	mask = Mat::zeros(imgSrc.rows, imgSrc.cols, CV_8UC1);
	int xSize = imgSrc.rows;
	int ySize = imgSrc.cols;
	for (int y = 0; y < ySize; y++)
	{
		for (int x = 0; x < xSize; x++)
		{
			Point p(x, y);
			int point[6] = { 0,0,0,0,0,0 };
			getGridPoint(0, p, point, 1, xSize, ySize);
			float probable = gridProbable.at<Vec3f>(point)[0];
			mask.at<uchar>(p.x, p.y) = (uchar)(probable * 255);
		}
	}

}

void Bilateral::run(Mat& mask) {

	mask.create(imgSrc.rows, imgSrc.cols, CV_8UC1);

	GCGraph<double> graph;//ͼ��

	constructGCGraph(graph);
	//getColor();
	estimateSegmentation(graph, mask);
	savePreImg("E:/Projects/OpenCV/DAVIS-data/examples/output/14.bmp", graph);
}