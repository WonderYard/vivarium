const mapNamedColor = (color: string): string => {
  switch (color.toLowerCase()) {
    case "black":
      return "#000000";
    case "silver":
      return "#c0c0c0";
    case "gray":
      return "#808080";
    case "white":
      return "#ffffff";
    case "maroon":
      return "#800000";
    case "red":
      return "#ff0000";
    case "purple":
      return "#800080";
    case "fuchsia":
      return "#ff00ff";
    case "green":
      return "#008000";
    case "lime":
      return "#00ff00";
    case "olive":
      return "#808000";
    case "yellow":
      return "#ffff00";
    case "navy":
      return "#000080";
    case "blue":
      return "#0000ff";
    case "teal":
      return "#008080";
    case "aqua":
      return "#00ffff";
    default:
      return color;
  }
};

export const colorToABGR = (color: string) => {
  color = mapNamedColor(color);

  const shorthandRegex = /^#?([a-f\d])([a-f\d])([a-f\d])$/i;
  color = color.replace(
    shorthandRegex,
    (_, r: string, g: string, b: string) => r + r + g + g + b + b
  );

  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(color);

  if (!result) {
    throw new Error("Bad color");
  }

  const r = parseInt(result[1], 16);
  const g = parseInt(result[2], 16);
  const b = parseInt(result[3], 16);

  return 0xff000000 | (b << 16) | (g << 8) | r;
};
