import rospy
import argparse
import cv2
import glob
import os
import time
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from realesrgan.archs.srvgg_arch import SRVGGNetCompact



   
   
class SuperReslt:
    
    def __init__(self):
        
        # 定义发布者
        self.pub = rospy.Publisher("/super_resolt_out", Image, queue_size=1)
        # 接受订阅话题，触发回调函数
        self.sub = rospy.Subscriber("/待接收图像话题名称", Image, callback)
        print("Waiting for acceptance...")
    
    
    def publish_image(self, pub, data, frame_id='base_link'):

	assert len(data.shape) == 3, 'len(data.shape) must be equal to 3.'
	self.header = Header(stamp=rospy.Time.now())
	self.header.frame_id = frame_id    
	self.msg = Image()
	self.msg.height = data.shape[0]
	self.msg.width = data.shape[1]
	self.msg.encoding = 'rgb8'
	self.msg.data = np.array(data).tostring()
	self.msg.header = self.header
	self.msg.step = self.msg.width * 1 * 3
	self.pub.publish(self.msg)
	
	
    def callback(self, data):

        print('Processing...')
        time0 = time.time()
        # 读入数据部分
        self.img = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
        # 计算输出
        self.output, _ = upsampler.enhance(self.img, outscale=args.outscale)
        # 发布结果
        self.publish_image(self.pub, self.output, frame_id='base_link')
        time1 = time.time()
        print('frame_rate = %f' % 1/(time1-time0))


def parse_args():
   
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument('-s', '--outscale', type=float, default=2, help='The final upsampling scale of the image')
    parser.add_argument('--suffix', type=str, default='out', help='Suffix of the restored image')
    parser.add_argument('-t', '--tile', type=int, default=0, help='Tile size, 0 for no tile during testing')
    parser.add_argument('--tile_pad', type=int, default=10, help='Tile padding')
    parser.add_argument('--pre_pad', type=int, default=0, help='Pre padding size at each border')
    #parser.add_argument('--face_enhance', action='store_true', help='Use GFPGAN to enhance face')
    parser.add_argument('--fp32', action='store_true', help='Use fp32 precision during inference. Default: fp16 (half precision).')
    parser.add_argument('--alpha_upsampler', type=str, default='realesrgan', help='The upsampler for the alpha channels. Options: realesrgan | bicubic')
    parser.add_argument('--ext', type=str, default='auto', help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs')
    args = parser.parse_args()
    return  parser.parse_args()
    

# 主函数
if __name__ == '__main__':
    
    # 初始化节点
    rospy.init_node('super_resolt_node', anonymous=True)

    # 加载全局参数
    args = parse_args()

    # 加载模型
    print('Loading Model...')
    # determine models according to model names
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    netscale = 2
    # determine model paths
    model_path = os.path.join('realesrgan/weights', 'RealESRGAN_x2plus.pth')
    # restorer
    upsampler = RealESRGANer(scale=netscale, model_path=model_path, model=model.cuda(), tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad, half=not args.fp32)

    # 定义对象
    super_resolt = SuperReslt()

    # 及时输出队列内容
    rospy.spin()
