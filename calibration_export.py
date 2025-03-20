from d435_test.calibration import take_photo,solve_pnp,calibrate_intrinsic

if __name__=="__main__":
    #take_photo(save_dir="./data/chessboard_images")
    # solve_pnp()
    matrix,dist=calibrate_intrinsic(chessboard_picpath="./data/chessboard_images",chessboard_size = (8, 5, 26.36),confirm=False)
    