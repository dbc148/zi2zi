from os.path import join
import os
from collections import defaultdict
import argparse
import cPickle as pickle
from PIL import Image, ImageChops
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import time

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def make_image(img, path):
	new_path = '/'.join(path.split('/')[:-1])
	makedir(new_path)
	img.save(path)

def trim(im):
	bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
	diff = ImageChops.difference(im, bg)
	diff = ImageChops.add(diff, diff, 2.0, -100)
	bbox = diff.getbbox()
	if bbox:
		return im.crop(bbox)

def embed_image(img,w,h):
	img_w, img_h = img.size
	if img_w > w:
		img = img.crop((0,0,w,img_h))
	if img_h > h:
		img = img.crop((0,0,img_w,h))
	img_w, img_h = img.size

	x = (w - img_w)/2
	y = (h - img_h)/2

	background = Image.new("RGBA", (w, h), (255, 255, 255))
	background.paste(img, (x,y))
	return background

def write_font(fnt, txt, w,h):
	temp_img = Image.new("RGBA", (1000,1000),(255,255,255))
	draw = ImageDraw.Draw(temp_img)
	text_x, text_y = fnt.getsize(txt)
	x = (w - text_x)/2
	y = (h - text_y)/2
	draw.text((x,y),txt,(0,0,0),font=fnt)
	
	trimmed_font_image = trim(temp_img)	
	
	trimmed_size = trimmed_font_image.size
	text_x = trimmed_size[0]
	text_y = trimmed_size[1]

	img = Image.new("RGBA",(w,h),(255,255,255))

	x = (w - text_x)/2
	y = (h - text_y)/2
	img.paste(trimmed_font_image, (x,y))
	return img


class Image_Container:
	#i might only need author, h, w, path, others just in case
	#h,w makes padding easier, as they are already known
	#path: for loading image during training (too many images to pre-load?)
	#author: for potential identification
	def __init__(self, author, h, w, word, form_id, image_path):
		self.image_path = image_path
		self.author = author
		self.form_id = form_id
		self.h = h
		self.w = w
		self.word = word

def process_words_file(file_path = 'letter_pairs.txt'):
	word_dict = defaultdict(int)
	print 'processing words file'
	word_file = open(file_path)
	word_lines = [line.strip()for line in word_file]
	images = []
	max_h = 0
	max_w = 0
	for idx, word in enumerate(word_lines):
		word_dict[word] += 1


	return word_dict


def write_example_images(word_dict, size=70, write_path = 'typed_words/', font_path = '/home/johnzz/dan/zi2zi/TALKTOTH.TTF', target_font_path = '/home/johnzz/dan/zi2zi/TALKTOTH.TTF', label='0'):
	fnt = ImageFont.truetype(font_path,size)
	target_fnt = ImageFont.truetype(target_font_path,size)
	words_2_fontpath = {}
	print 'finding correct width'

	width = 0
	max_idx = 0
	for idx, word in enumerate(word_dict):
		txt = word

		text_x, text_y = fnt.getsize(txt)
		if text_x > width and idx != 5333:
			width = text_x
			max_idx = idx


	
	print str(max_idx) + ' is the widest image with a width of: ' + str(width+4)

	print 'now creating combined images'
	width = width + 4
	height = 256
	width = 256
	height = 64
	idx = 0
	for word in word_dict:
		txt = word
		#img = Image.open('words/' + img_cont.image_path)
		example = Image.new("RGBA", (width * 2, height), (255, 255, 255))
		example.paste(write_font(fnt, txt, width, height), (width,0))
		example.paste(write_font(target_fnt, txt, width, height), (0, 0))
		make_image(example, join(write_path,label, label + '_' + txt + str(idx) + '.jpg'))
		if idx%500 == 0:
			print 'processed ' + str(idx) + ' images'
		idx += 1

parser = argparse.ArgumentParser(description='Processes images, converts font to images')
parser.add_argument('--src_font', dest='src_font', default = '/home/johnzz/dan/zi2zi/gbs.ttf', help='path of the source font')
parser.add_argument('--tgt_font', dest='tgt_font', default = '/home/johnzz/dan/zi2zi/Worstveld.otf', help='path of the target font')
parser.add_argument('--char_size', dest='char_size', type=int, default=150, help='character size')
parser.add_argument('--canvas_size', dest='canvas_size', type=int, default=256, help='canvas size')
parser.add_argument('--sample_dir', dest='sample_dir',default='typed_words/', help='directory to save examples')
parser.add_argument('--label', dest='label', type=int, default=0, help='label as the prefix of examples')
parser.add_argument('--x_offset', dest='x_offset', type=int, default=20, help='x offset')
parser.add_argument('--y_offset', dest='y_offset', type=int, default=20, help='y_offset')
args = parser.parse_args()

if __name__ == "__main__":
	word_dict = process_words_file()
	write_example_images(word_dict)
	with open('images.pickle', 'w') as f:  
		pickle.dump([images, max_w, max_h, word_dict], f)
