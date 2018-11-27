### hexo博客安装步骤
##### 安装git 和node.js
省略掉。我这里安装的版本为
```
$ git --version && node -v
git version 2.19.1.windows.1
v10.13.0
```
#### 安装hexo
```
npm install hexo -g
```
安装完成之后，输入hexo -v 检查hexo是否安装成功。

#### 启动项目
有两种方式，一种直接新建项目，一种，拿现在仓库项目直接跑起来。
- 新建空的项目
    输入 hexo init,初始化文件夹
    ```
    hexo init
    npm install
    ```
    生成页面
    ```
    hexo g
    ```
    启动服务
    ```
    hexo s
    ```
- 直接使用仓库代码创建
    ```
    git clone git@github.com:hobbitmr/hexoblog.git
    npm install
    hexo g && hexo s
    ```
#### 安装编辑器 HexoEditor
 [github地址](https://github.com/zhuzhuyule/HexoEditor)
 安装步骤
 ```
 //if use Windows:
npm config set prefix "C:/Program Files/nodejs/npm_global"
npm config set cache "C:/Program Files/nodejs/npm_cache" 

//if use Linux\Mac:
npm config set prefix "~/nodejs/npm_global"
npm config set cache "~/nodejs/npm_cache" 

//If In China, China, China, you can set mirror to speed up !
//如果你为了加快速度用了淘宝源，使用安装完成之后，尽量换回原来的源。淘宝源缺了一些东西。会导致你后面有很多莫名其妙的错误
npm config set registry "https://registry.npm.taobao.org/"
//切换原来的源
npm config set registry https://registry.npmjs.org

git clone https://github.com/zhuzhuyule/HexoEditor.git
cd HexoEditor
npm install
npm start
 ```
为了更加方便启动，我们可以制作一个bat脚本。来快速启动他

start.bat
```
@echo off
cd /d %~dp0
npm start
```
将这个脚本放在HexoEditor的目录下面。然后创建快捷方式，修改快捷方式的图标
![](https://i.loli.net/2018/11/27/5bfcfbcc55c4b.png)
这样子就打工告成。。。
![](https://i.loli.net/2018/11/27/5bfcfbcc2bb7a.png)




