from os.path import join
import os
from collections import defaultdict
import argparse
import cPickle as pickle
from PIL import Image
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
	img = Image.new("RGBA",(w,h),(255,255,255))
	draw = ImageDraw.Draw(img)
	text_x, text_y = fnt.getsize(txt)
	x = (w - text_x)/2
	y = (h - text_y)/2
	draw.text((x,y),txt,(0,0,0),font=fnt)
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

def process_words_file(file_path = 'words.txt', form_path = 'forms.txt'):
	word_dict = defaultdict(int)
	form_file = open(form_path)
	form_lines = [line.strip().split() for line in form_file]
	form_authors = {}
	print 'processing forms file'
	for line in form_lines:
		form_authors[line[0]] = line[1]
	print 'finished processing forms file, now processing words file'
	word_file = open(file_path)
	word_lines = [line.strip().split() for line in word_file]
	images = []
	max_h = 0
	max_w = 0
	for idx, line in enumerate(word_lines):
		if line[1] != 'ok':
			continue
		form_id = '-'.join(line[0].split('-')[:2])
		author = form_authors[form_id]
		image_path = join(form_id.split('-')[0], form_id, line[0]+'.png')
		w = int(line[5])
		h = int(line[6])
		word = ' '.join(line[8:])
		word_dict[word] += 1
		if w > max_w:
			max_w = w
		if h > max_h:
			max_h = h
		images.append(Image_Container(author, h, w, word, form_id, image_path))
	print max_w, max_h, 'max width, max height'

	return images, max_w, max_h, word_dict

def write_all_fonts(word_dict, x_offset, y_offset, size=150, write_path = 'typed_words/', font_path = '/home/johnzz/dan/zi2zi/TALKTOTH.TTF'):
	fnt = ImageFont.truetype(font_path,size)
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
			# width = width + 4
			# height = 256
			# img = Image.new("RGBA",(width,height),(255,255,255))
			# draw = ImageDraw.Draw(img)
			# text_x, text_y = fnt.getsize(txt)
			# x = (width - text_x)/2
			# y = (height - text_y)/2
			# draw.text((x,y),txt,(0,0,0),font=fnt)
			# img.save(write_path + str(idx) + '.png')


	print str(max_idx) + ' is the widest image with a width of: ' + str(width+4)

	print 'now creating font images'
	width = width + 4
	height = 256

	for idx, word in enumerate(word_dict):
		txt = word
		text_x, text_y = fnt.getsize(txt)
		#width = 10*256
		img = Image.new("RGBA",(width,height),(255,255,255))
		draw = ImageDraw.Draw(img)
		text_x, text_y = fnt.getsize(txt)
		x = (width - text_x)/2
		y = (height - text_y)/2
		draw.text((x,y),txt,(0,0,0),font=fnt)
		img.save(write_path + str(idx) + '.png')
		words_2_fontpath[word] = write_path + str(idx) + '.png'
		if idx%500 == 0:
			print 'processed ' + str(idx) + ' images'
	return words_2_fontpath


def write_example_images(images, word_dict, max_width, max_height, size=150, write_path = 'typed_words/', font_path = '/home/johnzz/dan/zi2zi/TALKTOTH.TTF', target_font_path = '/home/johnzz/dan/zi2zi/TALKTOTH.TTF', label='0'):
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
	if width < max_w:
		width = max_w
	if height < max_h:
		height = max_h
	width = 1024
	height = 256
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
	images, max_w, max_h, word_dict = process_words_file()
	#words2font = write_all_fonts(word_dict, args.x_offset, args.y_offset, args.char_size)
	write_example_images(images, word_dict, max_w, max_h, font_path = args.src_font, target_font_path = args.tgt_font)
	with open('images.pickle', 'w') as f:  
		pickle.dump([images, max_w, max_h, word_dict], f)
