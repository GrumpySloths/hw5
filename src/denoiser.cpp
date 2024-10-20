#include "denoiser.h"
#include<assert.h>

Denoiser::Denoiser() : m_useTemportal(false) {}

void Denoiser::Reprojection(const FrameInfo &frameInfo) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    Matrix4x4 preWorldToScreen =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 1];
    Matrix4x4 curWorldToScreen = frameInfo.m_matrix[frameInfo.m_matrix.size() - 1];
    Matrix4x4 preWorldToCamera =
        m_preFrameInfo.m_matrix[m_preFrameInfo.m_matrix.size() - 2];
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Reproject
            Float3 curPosition = frameInfo.m_position(x, y);
            // std::cout<<"preWorldToCamera:\n"<<preWorldToCamera<<std::endl;
            // std::cout<<"preWorldToScreen:\n"<<preWorldToScreen<<std::endl;
            // std::cout<<"curWorldToScreen:\n"<<curWorldToScreen<<std::endl;
            Float3 position_camera=preWorldToCamera(curPosition,Float3::Point);
            //获取相机坐标系下的深度值并和深度图进行比较，看是否在屏幕上
            // std::cout<<"position_camera:"<<position_camera<<std::endl;
            // std::cout<<"position"<<curPosition<<std::endl;
            // std::cout<<"pre depth:"<<m_preFrameInfo.m_depth(x,y)<<std::endl;
            //判断是否在屏幕上
            if(position_camera.z<=m_preFrameInfo.m_depth(x,y)){
                Float3 pre_screen = preWorldToScreen(curPosition, Float3::Point);
                // std::cout<<"pre_screen:"<<pre_screen<<std::endl;
                int pre_x = pre_screen.x;
                int pre_y = pre_screen.y;
                int pre_id=m_preFrameInfo.m_id(pre_x,pre_y);
                int cur_id=frameInfo.m_id(x,y);
                // std::cout<<"pre_id:"<<pre_id<<" cur_id:"<<cur_id<<std::endl;
                //判断是否为同一物体
                if(pre_id==cur_id){
                    m_valid(x, y) = true;
                    m_misc(x, y) = m_accColor(pre_x, pre_y);
                    continue;
                }

            }
            //失效处理
            m_valid(x, y) = false;
            m_misc(x, y) = Float3(0.0f);
        }
    }
    std::swap(m_misc, m_accColor);
}

void Denoiser::TemporalAccumulation(const Buffer2D<Float3> &curFilteredColor) {
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    int kernelRadius = 3;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if(m_valid(x,y)==false){
                m_misc(x, y) = curFilteredColor(x, y);
                continue;
            }
            // TODO: Temporal clamp
            Float3 color = m_accColor(x, y);
            //计算 以(x,y)为中心，kernelRadius为半径的均值和标准差
            int x0 = std::max(0, x - kernelRadius);
            int x1=std::min(width,x+kernelRadius);
            int y0 = std::max(0, y - kernelRadius);
            int y1=std::min(height,y+kernelRadius);
            Float3 sum=0;
            int count=0;
            for(int i=x0;i<x1;i++){
                for(int j=y0;j<y1;j++){
                    sum+=curFilteredColor(i,j);
                    count++;
                }
            }
            Float3 mean=sum/count;
            //计算方差
            Float3 variance=0;
            for(int i=x0;i<x1;i++){
                for(int j=y0;j<y1;j++){
                    variance+=Sqr(curFilteredColor(i,j)-mean);
                }
            }
            Float3 sigma=SafeSqrt(variance/count);

            color=Clamp(color,mean-sigma*m_colorBoxK,mean+sigma*m_colorBoxK);

            // TODO: Exponential moving average
            float alpha=0.2f;
            m_misc(x, y) = Lerp(color, curFilteredColor(x, y), alpha);
        }
    }
    std::swap(m_misc, m_accColor);
}
float D_normal(const Float3&n1,Float3&n2){
    return Sqr(SafeAcos(Dot(n1,n2)));
}

float D_plane(const Float3&ni,const Float3&pi,const Float3&pj){
    if(pi==pj) return 0;
    return Sqr(Dot(ni,Normalize(pj-pi)));
}

Buffer2D<Float3> Denoiser::Filter(const FrameInfo &frameInfo) {
    int height = frameInfo.m_beauty.m_height;
    int width = frameInfo.m_beauty.m_width;
    Buffer2D<Float3> filteredImage = CreateBuffer2D<Float3>(width, height);
    int kernelRadius = 32;
    // float weightSum = 0;
#pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // TODO: Joint bilateral filter
            filteredImage(x, y) = frameInfo.m_beauty(x, y);
            //确定滤波的边界
            int x0 = std::max(0, x - kernelRadius/2);
            int x1 = std::min(width, x + kernelRadius/2);
            int y0 = std::max(0, y - kernelRadius/2);
            int y1 = std::min(height, y + kernelRadius/2);
            //计算滤波核的权重
            float weightSum=1;
            for(int i=x0;i<x1;i++){
                for(int j=y0;j<y1;j++){
                    //如果是自己则跳过
                    if(i==x&&j==y){
                        continue;
                    }
                    //计算空间权重
                    Float3 pi = frameInfo.m_position(x,y);
                    Float3 pj = frameInfo.m_position(i,j);
                    float d_coord=SqrDistance(pi,pj);
                    // float d_coord=Sqr(i-x)+Sqr(j-y);
                    float w_coord=exp(-d_coord/(2*m_sigmaCoord*m_sigmaCoord));
                    //计算颜色权重
                    Float3 ci = frameInfo.m_beauty(x,y);
                    Float3 cj = frameInfo.m_beauty(i,j);
                    float d_color=SqrDistance(ci,cj);
                    float w_color=exp(-d_color/(2*m_sigmaColor*m_sigmaColor));
                    //计算法线权重
                    Float3 ni = frameInfo.m_normal(x,y);
                    Float3 nj = frameInfo.m_normal(i,j);
                    float d_normal=D_normal(ni,nj);
                    float w_normal=exp(-d_normal/(2*m_sigmaNormal*m_sigmaNormal));
                    //计算平面权重
                    float d_plane=D_plane(ni,pi,pj);
                    float w_plane=exp(-d_plane/(2*m_sigmaPlane*m_sigmaPlane));
                    //计算最终权重
                    float w=w_coord*w_color*w_normal*w_plane;
                    weightSum+=w;
                    //计算滤波后的颜色
                    filteredImage(x,y)+=frameInfo.m_beauty(i,j)*w;
                }
            }
            assert(weightSum<=kernelRadius*kernelRadius);            
            // std::cout<<"x:"<<x<<" y:"<<y<<" weightSum:"<<weightSum<<std::endl;
            filteredImage(x,y)/=weightSum;
        }
    }
    return filteredImage;
}

void Denoiser::Init(const FrameInfo &frameInfo, const Buffer2D<Float3> &filteredColor) {
    m_accColor.Copy(filteredColor);
    int height = m_accColor.m_height;
    int width = m_accColor.m_width;
    m_misc = CreateBuffer2D<Float3>(width, height);
    m_valid = CreateBuffer2D<bool>(width, height);
}

void Denoiser::Maintain(const FrameInfo &frameInfo) { m_preFrameInfo = frameInfo; }

Buffer2D<Float3> Denoiser::ProcessFrame(const FrameInfo &frameInfo) {
    // Filter current frame
    Buffer2D<Float3> filteredColor;
    filteredColor = Filter(frameInfo);

    // Reproject previous frame color to current
    if (m_useTemportal) {
        Reprojection(frameInfo);
        TemporalAccumulation(filteredColor);
    } else {
        Init(frameInfo, filteredColor);
    }

    // Maintain
    Maintain(frameInfo);
    if (!m_useTemportal) {
        m_useTemportal = true;
    }
    return m_accColor;
}
