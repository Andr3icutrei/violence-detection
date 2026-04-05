import { Component } from '@angular/core';
import { Topbar } from '../../topbar/topbar';
import { InspectVideos } from './inspect-videos/inspect-videos';

@Component({
  selector: 'app-dashboard-page',
  imports: [InspectVideos],
  templateUrl: './dashboard-page.html',
  styleUrl: './dashboard-page.css',
})
export class DashboardPage {}
