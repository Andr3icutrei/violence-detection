import { ComponentFixture, TestBed } from '@angular/core/testing';

import { VerifyAccountSuccess } from './verify-account-success';

describe('VerifyAccountSuccess', () => {
  let component: VerifyAccountSuccess;
  let fixture: ComponentFixture<VerifyAccountSuccess>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [VerifyAccountSuccess],
    }).compileComponents();

    fixture = TestBed.createComponent(VerifyAccountSuccess);
    component = fixture.componentInstance;
    await fixture.whenStable();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
