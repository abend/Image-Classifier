=head1 NAME

Image::Classifier - A silhouette-based image classifier.

=head1 SYNOPSIS

  use Image::Classifier;
  my $classifier = Image::Classifier->new({training_dir => "/path/to/known/types",
                                           work_dir => "/writable/directory"});
  my ($type, $confidence) = $classifier->classify($candidate_filename);

=head1 DESCRIPTION

Classify an image by silhouette into a category, based on sets of
images of known categories.

It requires a training directory with several subdirectories, one for
each category.  The subdirectory name is the category name.  It
contains images whose silhouettes match that category.  The candidate
image is compared to each of the training images by matching corner
features on the silhouettes.

=head1 USAGE

  use Image::Classifier;
  my $classifier = Image::Classifier->new({training_dir => "/path/to/known/types",
                                           work_dir => "/writable/directory"});
  my ($type, $confidence) = $classifier->classify($candidate_filename);

=head1 BUGS

You tell me.

=head1 AUTHOR

    Sasha Kovar
    CPAN ID: ABEND
    sasha-cpan@arcocene.org

=head1 COPYRIGHT

This program is free software; you can redistribute
it and/or modify it under the same terms as Perl itself.

The full text of the license can be found in the
LICENSE file included with this module.

=head1 SEE ALSO

Image::EdgeDetect Image::CornerDetect

=cut

package Image::Classifier;
use strict;
use warnings;
use 5.010;

use File::Path;
use File::Basename;
use YAML;
use List::AllUtils qw(max min);
use Image::Magick;
use Image::EdgeDetect;
use Image::CornerDetect;
use Data::Dumper;

BEGIN {
    use Exporter ();
    use vars qw($VERSION @ISA @EXPORT @EXPORT_OK %EXPORT_TAGS);
    $VERSION     = '0.6';
    @ISA         = qw(Exporter);
    @EXPORT      = qw();
    @EXPORT_OK   = qw();
    %EXPORT_TAGS = ();
}


=head2 my $classifier = Image::Classifier->new(\%args)

Create a new classifier, passing in parameters as follows:

  training_dir - Path to a directory containing training images,
      organized by subdirectory.

  work_dir - Path to a writable directory for placing working files.
      Defaults to training_dir.

  corner_params - Hashref of parameters to pass to the corner
      detector.  See Image::CornerDetect for details.

  force_refresh - If true, ignore cached corner data files and
      regenerate.

  image_size - Width and height of the generated test images.
      Defaults to 200px.

  image_border - Border size around the contents of the generated images.
      Defaults to 4px.

  match_radius - Maximum distance between candidate points to consider
      matched.  Defaults to 1/20th the image_size.

  debug_images - Write out images from intermediate stages in the
      classification process to the work_dir.  Set to 1 to write a
      composite of corners and edges over the original image.  Set to
      2 to also write separate silhouette, edge, and corner images.
      Set to 3 to add a clipped version of the original image.
      Default is 0 (no images).

=cut

sub new {
  my ($this, $args) = @_;
  my $class = ref($this) || $this;

  my %args = ref($args) eq 'HASH' ? %$args : ();

  my $self =
  {
   training_dir  => $args{training_dir},
   work_dir      => $args{work_dir} || $args{training_dir},
   force_refresh => $args{force_refresh},
   image_size    => $args{image_size} || 200,
   image_border  => $args{image_border} || 4,
   debug_images  => $args{debug_images},
   corner_params => $args{corner_params},
  };

  $self->{match_radius} = $args{match_radius} || ($self->{image_size} / 20);

  bless $self, $class;

  $self->init();

  return $self;
}

=head2 ($type, $confidence) = $classifier->classify($candidate_filename);

Classify the input image, returning the closest match type and confidence level.

=cut

sub classify {
  my ($self, $filename) = @_;

  my ($type, $confidence, $closest);

  my @corners = $self->getCorners($filename, 1);

  # find closest match.  brute force.  could try some k-nearest
  # neighbors or something fancy someday.
  my $td = $self->{training_data};
  for my $k (keys %$td) {
    my @samples = @{$$td{$k}};

    for my $s (@samples) {
      my $samplefile = shift @$s;
      my $score = $self->score(\@corners, $s);
      if (!$confidence or $score > $confidence) {
        $type = $k;
        $confidence = $score;
        $closest = $samplefile;
      }
    }
  }

  return $type, $confidence, $closest;
}

# return how close the two sets of corners are matched, as a number
# between 0 and 1.
sub score {
  my ($self, $test, $candidate) = @_;

  return 0 unless @$test && @$candidate;

  my $distsq = $self->{match_radius} ** 2;
  my $score = 0;

  for my $sa (@$test) {
    my $closest = 0;
    for my $sb (@$candidate) {
      next unless $sa && $sb;

      my $distance = ($$sa[0] - $$sb[0])**2 + ($$sa[1] - $$sb[1])**2;
      my $scale = $distance / $distsq;
      my $sc = $scale < 1 ? 1 - $scale : 0;
      $closest = max($sc, $closest);
    }

    $score += $closest;
  }

  my $count = max(scalar @$test, scalar @$candidate);
  $score /= $count;

  return $score;
}

# scan training dir, loading (and generating where necessary) the
# classification data.
sub init {
  my ($self) = @_;

  my %data;

  my $tdir = $self->{training_dir};

  for my $typedir (sort grep { -d $_ } glob("$tdir/*")) {
    my $type = basename($typedir);
    #say "loading type '$type' from $typedir";

    my @sets = ();

    for my $img (sort grep { -f $_ } glob("$typedir/*")) {
      my @corners = $self->getCorners($img);

      push @sets, [basename($img), @corners];
    }

    $data{$type} = [@sets];
  }

  $self->{training_data} = \%data;
}

sub getCorners {
  my ($self, $filename, $dont_cache) = @_;

  # try cached
  my $cf = $self->cornerFile($filename);
  return $self->loadCorners($cf) if (!$self->{force_refresh} and
                                     -f $cf
                                     and (stat($cf))[9] > (stat($filename))[9]);

  # make silhouette
  my ($sil, $orig) = $self->makeSilhouette($filename);
  $self->writeDebugImage($sil, $filename, "sil", 2);

  # make edge mask
  my $edetector = Image::EdgeDetect->new({
                                          #low_threshold => 2.5,
                                          #high_threshold => 7.5,
                                          kernel_radius => 1.5,#2.0,
                                          kernel_width => 3,#16,
                                         });
  my $edge = $edetector->process($sil);
  $self->writeDebugImage($edge, $filename, "edge", 2);

  # detect corners
  my $cdetector = Image::CornerDetect->new($self->{corner_params});
  my @corners = $cdetector->process($edge);

  if ($self->{debug_images}) {
    # mark edges and corners
    $orig->Modulate(brightness=>20);
    $orig->Composite(image=>$edge, compose=>'Lighten');

    my $w = 2;
    for my $corner (@corners) {
      my ($x, $y) = @$corner;
      $orig->Draw(fill=>'red', primitive=>'rectangle',
                  points=>sprintf("%d,%d %d,%d", $x-$w, $y-$w, $x+$w, $y+$w));
    }

    $self->writeDebugImage($orig, $filename, "corner", 2);
  }

  $self->saveCorners($cf, \@corners) unless $dont_cache;

  return @corners;
}

sub cornerFile {
  my ($self, $img) = @_;

  return $self->workFile($img, undef, "corners")
}

sub loadCorners {
  my ($self, $cornerfile) = @_;

  open(my $fh, '<', $cornerfile) or die "can't open corner file '$cornerfile' for read";
  my @corners = Load(join("\n",<$fh>));
  close $fh;

  #say "loading corner file $cornerfile got ".@corners;

  return @corners;
}

sub saveCorners {
  my ($self, $cornerfile, $corners) = @_;

  open(my $fh, '>', $cornerfile) or die "can't open corner file '$cornerfile' for write";
  print $fh Dump(@$corners);
  close $fh;
}

# return a Image::Magick image.
sub makeSilhouette {
  my ($self, $filename) = @_;

  my $image = Image::Magick->new;
  #$image->Set(density => 288);
  $image->Set(density => 72*8);
  my $status = $image->Read($filename);
  warn $status if $status;

  my $s = $self->{image_size};
  my $border = $self->{image_border};
  my $s2 = $s - ($border*2);
  my $size = "${s}x$s";
  my $scaleSize = "${s2}x$s2";

  $image->Set(colorspace => 'RGB');
  #$image->Profile(name => "$Settings::ASSETDIR/icc/sRGB_v4_ICC_preference.icc");
  $image->Set(depth => 8);
  $image->Resize(geometry=>$scaleSize, blur => .8);
  $image->Set(alpha => 'Set');
  $image->Clip();
  $image->Negate(channel => 'Alpha');

  my $tmpfile = "/tmp/tmp$$.png";
  $image->Write($tmpfile);

  $image = Image::Magick->new;
  $status = $image->Read($tmpfile);
  warn $status if $status;
  unlink($tmpfile);

  $image->Negate(channel => 'Alpha');

  if (transparentAmount($image) == 0) {
    # flood fill
    $image->Set(fill => 'none');
    my @pixels = $image->GetPixel(x=>0,y=>0);
    my $color = sprintf("#%0.2X%0.2X%0.2X", map { $_ * 255 } @pixels);
    $image->Border(geometry => '1x1', fill => $color);
    $image->MatteFloodfill(geometry => '0x0', fuzz => '2%');
  }

  $self->writeDebugImage($image, $filename, "clip", 3);

  my $white = Image::Magick->new;
  $white->Set(size=>$scaleSize);
  $white->ReadImage('xc:white');

  my $pad = Image::Magick->new(size => $size.'x'.$size);
  $pad->ReadImage("xc:white");
  $pad->Composite(image=>$image, compose=>'CopyOpacity', gravity=>'Center');

  my $black = Image::Magick->new;
  $black->Set(size=>$size);
  $black->ReadImage('xc:black');
  $black->Composite(image=>$pad, compose=>'SrcOver', gravity=>'Center');
  $black->Quantize(colors=>2);

  my $orig = Image::Magick->new(size => $size.'x'.$size);
  $orig->ReadImage("xc:white");
  $orig->Composite(image=>$image, compose=>'SrcOver', gravity=>'Center');

  return $black, $orig;
}

# return the amount of the image that is transparent, from 0-1.
# requires the image to be saved with an alpha channel first, ie clip
# paths aren't considered.
sub transparentAmount {
  my $image = shift;

  my @h = $image->Histogram();
  my ($trans, $total) = (0, 0);
  for (my $i = 0; $i < @h; $i += 5) {
    my ($r, $g, $b, $a, $cnt) = @h[$i..$i+4];
    if ($a > 65530) {
      $trans += $cnt;
    }
    $total += $cnt;
  }

  return $trans / $total;
}

sub writeDebugImage {
  my ($self, $img, $file, $type, $level) = @_;

  return unless $self->{debug_images} >= $level;

  $img->Write($self->workFile($file, $type, "gif"));
}

# convert an image file path to a work file for that image.
sub workFile {
  my ($self, $file, $type, $ext) = @_;

  my ($base, $dir, $suf) = fileparse($file, qr{\..*$});

  $dir =~ s/$self->{training_dir}/$self->{work_dir}/;

  mkpath($dir);

  return sprintf("%s/%s%s.%s", $dir, $base, ($type ? "-$type" : ''), $ext);
}

1;

