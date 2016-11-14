#include <iostream>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include "Bilateral.h"

using namespace cv;


Mat g_imgSrc, g_imgShow; //原图与展示图
std::vector<Point> g_forePts, g_backPts; //vector相当于数组，分别存储前景点和背景点
bool IsLeftDown = false, IsRightDown = false;  //定义了两个假
Point currentPoint, nextPoint;

static void OnMouse(int even, int x, int y, int, void*) //调用鼠标，总地来说就是收集前景节点和背景节点保存在g_fore/backPts中，并实时显示
{
	if (even == CV_EVENT_LBUTTONDOWN) //鼠标左键点击
	{
		currentPoint = Point(x, y);
		g_forePts.push_back(currentPoint);
		IsLeftDown = true;
		return;
	}
	if (IsLeftDown&&even == CV_EVENT_MOUSEMOVE)//鼠标左键按着没松还在动
	{
		nextPoint = Point(x, y);
		g_forePts.push_back(nextPoint);
		line(g_imgShow, currentPoint, nextPoint, Scalar(255, 0, 0), 2); // 255.0.0是蓝色  10是线条宽度，类似于把鼠标点为中心10单位半径内的点归为point
		currentPoint = nextPoint;
		imshow("原图像", g_imgShow);
		return;
	}
	if (IsLeftDown&&even == CV_EVENT_LBUTTONUP) //松了
	{
		IsLeftDown = false;
		return;
	}

	if (even == CV_EVENT_RBUTTONDOWN) //按下右键
	{
		currentPoint = Point(x, y);
		g_backPts.push_back(currentPoint);
		IsRightDown = true;
		return;
	}
	if (IsRightDown&&even == CV_EVENT_MOUSEMOVE)//划线
	{
		nextPoint = Point(x, y);
		g_backPts.push_back(nextPoint);
		line(g_imgShow, currentPoint, nextPoint, Scalar(0, 0, 255), 2);
		currentPoint = nextPoint;
		imshow("原图像", g_imgShow);
		return;
	}
	if (IsRightDown&&even == CV_EVENT_RBUTTONUP)//松了
	{
		IsRightDown = false;
		return;
	}
}


int main() {
	g_forePts.clear();
	g_backPts.clear();
	g_imgSrc = imread("image/00001.jpg");//清空前景背景点，导入图片
	g_imgSrc.copyTo(g_imgShow);//备份一份
	namedWindow("原图像");
	setMouseCallback("原图像", OnMouse);//鼠标调用函数
	Mat mask, lastImg;//这一步？
	while (1)
	{
		imshow("原图像", g_imgShow);
		int t = waitKey();
		if (t == 27) break; //27就是esc,随时生效

		char c = (char)t;
		if (c == 's')   //键盘输入S实现分割 点那个分割画面 cmd啥事也干不了
		{
			Bilateral bilateral(g_imgSrc);
			bilateral.InitGmms(g_forePts, g_backPts);
			bilateral.run();
		}
	}
	mask.release();
	lastImg.release();
	g_imgSrc.release();
	g_imgShow.release();
	destroyAllWindows();

	return 0;
}